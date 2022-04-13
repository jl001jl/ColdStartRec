# -*- coding: UTF-8 -*-
"""
@Project ：EBSNEventRec
@File ：features
@Author ：Jingyu Jia
@Email: 1258660442@qq.com
@Date ：19:10
"""
import math

import pandas as pd
import tqdm
import numpy as np
from ebsnrec.data.utils import binary_search


def num_users_has_attend(sample_df: pd.DataFrame, **kwargs):
    event_df = kwargs["event_df"]
    res = []
    for rec_t, e_id in tqdm.tqdm(zip(sample_df["rec_time"], sample_df["event_id"]), total=len(sample_df),
                                 desc="parse feature num_users_has_attend"):
        idx = binary_search(event_df["attendance_times"][e_id], rec_t)
        res.append(idx + 1)
    sample_df["num_users_has_attend"] = pd.Series(res, dtype=int)


def user_join_event_group(sample_df: pd.DataFrame, **kwargs):
    user_df = kwargs["user_df"]
    event_df = kwargs["event_df"]
    res = []
    for u_id, e_id in tqdm.tqdm(zip(sample_df["user_id"], sample_df["event_id"]), total=len(sample_df),
                                desc="parse feature user_join_event_group"):
        e_group_id = event_df["group_id"][e_id]
        res.append(e_group_id in user_df["group_ids"][u_id])

    sample_df["user_join_event_group"] = pd.Series(res, dtype=int)


def user_event_topic_overlap(sample_df: pd.DataFrame, **kwargs):
    user_df = kwargs["user_df"]
    event_df = kwargs["event_df"]
    res = []
    for u_id, e_id in tqdm.tqdm(zip(sample_df["user_id"], sample_df["event_id"]), total=len(sample_df),
                                desc="parse feature  user_event_topic_overlap"):
        u_set = user_df["topic_ids"][u_id]
        e_set = event_df["topic_ids"][e_id]
        inter = len(e_set & u_set)
        outer = len(e_set | u_set)
        if inter == 0 or outer == 0:
            res.append(0.0)
        else:
            res.append(inter/outer)


    sample_df["user_event_topic_overlap"] = pd.Series(res, dtype=float)



def user_has_attend_event(sample_df: pd.DataFrame, **kwargs):
    event_df = kwargs["event_df"]
    res = []
    for n, e_id in tqdm.tqdm(zip(sample_df["num_users_has_attend"], sample_df["event_id"]), total=len(sample_df),
                                desc="parse feature user_has_attend_event"):
        user_list = event_df["attendance_ids"][e_id][max(0,n-20):n]
        res.append("^".join(user_list))

    sample_df["user_has_attend_event"] = pd.Series(res, dtype=str)



def context_similarity(sample_df: pd.DataFrame, **kwargs):
    user_df = kwargs["user_df"]
    event_df = kwargs["event_df"]
    res_social = []
    res_content = []
    res_time = []
    res_location = []
    user_joined_events = []
    for u_id, e_id, rec_t in tqdm.tqdm(zip(sample_df["user_id"], sample_df["event_id"], sample_df["rec_time"]),
                                       total=len(sample_df),
                                       desc="parse feature context_similarity"):
        n_events_user_attendance = binary_search(user_df["joined_event_times"][u_id], rec_t) + 1
        if n_events_user_attendance == 0:
            for lis in [res_content, res_social, res_time, res_location, user_joined_events]:
                lis.append(None)
        else:
            history_attendance_events = user_df["joined_event_ids"][u_id][
                                        max(0, n_events_user_attendance - 20):n_events_user_attendance]
            res_social.append(calculate_context_social(u_id=u_id, e_id=e_id, event_df=event_df, user_df=user_df,
                                                       history_attendance_events=history_attendance_events))
            res_location.append(calculate_context_location(u_id=u_id, e_id=e_id, event_df=event_df, user_df=user_df,
                                                           history_attendance_events=history_attendance_events))
            res_time.append(calculate_context_time(u_id=u_id, e_id=e_id, event_df=event_df, user_df=user_df,
                                                           history_attendance_events=history_attendance_events))
            res_content.append(calculate_context_content(u_id=u_id, e_id=e_id, event_df=event_df, user_df=user_df,
                                                   history_attendance_events=history_attendance_events))
            user_joined_events.append("^".join(history_attendance_events))

    sample_df["context_social_similarity"] = pd.Series(res_social, dtype=float)
    sample_df["context_location_similarity"] = pd.Series(res_location, dtype=float)
    sample_df["context_time_similarity"] = pd.Series(res_time, dtype=float)
    sample_df["context_content_similarity"] = pd.Series(res_content, dtype=float)
    sample_df["user_joined_events"] = pd.Series(user_joined_events, dtype=str)


def calculate_context_social(**kwargs):
    event_df = kwargs["event_df"]
    e_id = kwargs["e_id"]
    history_attendance_events = kwargs["history_attendance_events"]
    target_group = event_df["group_id"][e_id]

    hit = sum([1 for x in history_attendance_events if event_df["group_id"][x] == target_group])
    return hit / len(history_attendance_events)


def calculate_context_location(**kwargs):
    event_df = kwargs["event_df"]
    e_id = kwargs["e_id"]
    history_attendance_events = kwargs["history_attendance_events"]
    e_lat,e_lon = event_df["lat"][e_id],event_df["lon"][e_id]
    mean_lat = sum([event_df["lat"][x] for x in history_attendance_events])/len(history_attendance_events)
    mean_lon = sum([event_df["lon"][x] for x in history_attendance_events])/len(history_attendance_events)
    distance = math.sqrt((mean_lat-e_lat)**2 + (mean_lon-e_lon)**2)
    return distance


def calculate_context_time(**kwargs):
    event_df = kwargs["event_df"]
    e_id = kwargs["e_id"]
    history_attendance_events = kwargs["history_attendance_events"]
    e_hour = event_df["hour"][e_id]
    e_weekday = event_df["weekday"][e_id]

    hour_similarity =  sum([1 for x in history_attendance_events if event_df["hour"][x] == e_hour]) / len(history_attendance_events)
    weekday_similarity =  sum([1 for x in history_attendance_events if event_df["weekday"][x] == e_weekday]) / len(history_attendance_events)

    return 0.5*hour_similarity + 0.5*weekday_similarity


def calculate_context_content(**kwargs):
    event_df = kwargs["event_df"]
    e_id = kwargs["e_id"]
    history_attendance_events = kwargs["history_attendance_events"]
    vec_u = np.array(event_df["description"][history_attendance_events].to_list()).mean(axis=0)
    vec_e = np.array(event_df["description"][e_id])

    return np.dot(vec_u,vec_e)