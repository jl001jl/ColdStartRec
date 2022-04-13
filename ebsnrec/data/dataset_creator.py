import gc
import logging
import logging
import os.path
import sqlite3 as sql
from datetime import timedelta, timezone
import numpy as np
import pandas as pd
import tqdm
from dateutil import parser
from intervaltree import IntervalTree
from random import shuffle
from ebsnrec.data.utils import convert_df_cols_to_dtype,clean_text
from ebsnrec.data.mining_features import *
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


class DatasetCreator(object):
    def __init__(self,
                 raw_dataset_file: str,
                 sava_dir: str,
                 use_features=[],
                 tz_offset: int = 0,
                 begin_time: str = "2017-06-01",
                 end_time: str = "2018-01-01",
                 event_min_attend=5,
                 event_max_attend=50,
                 user_min_attend=5,
                 train_pr=0.9,
                 val_pr=0.05,
                 test_pr=0.05,
                 max_sample_per_event=10,
                 min_neg_per_pos=5,
                 max_neg_per_pos=20,
                 cold_start_reserve_length=5,
                 eval_max_neg_per_pos=100,
                 num_workers=6,
                 **kwargs):
        self.raw_dataset_file = raw_dataset_file
        self.save_dir = sava_dir
        self.use_features = use_features
        self.tz_info = timezone(timedelta(hours=tz_offset))
        self.begin_time = parser.parse(begin_time)
        self.end_time = parser.parse(end_time)
        self.event_min_attend = event_min_attend
        self.event_max_attend = event_max_attend
        self.user_min_attend = user_min_attend
        self.train_pr = train_pr
        self.val_pr = val_pr
        self.test_pr = test_pr
        self.max_sample_per_event = max_sample_per_event
        self.min_neg_per_pos = min_neg_per_pos
        self.max_neg_per_pos = max_neg_per_pos
        self.cold_start_reserve_length = cold_start_reserve_length
        self.eval_max_neg_per_pos = eval_max_neg_per_pos
        self.num_works = num_workers

        self.cache_file = os.path.join(self.save_dir, "cache.pkl")
        self.desc_file = os.path.join(self.save_dir, "dataset_desc.json")

        self.event_df = None
        self.user_df = None
        self.time_query_tree = None
        self.events = None
        self.users = None
        self.num_events = None
        self.num_users = None
        self.user2events = None

    def create_dataset(self):
        if self.check_can_skip():
            logging.warning(f"skip create data {self.raw_dataset_file}")
            return
        self.load_raw_dataset()


    def load_raw_dataset(self):
        logging.info(f"start load raw data from {self.raw_dataset_file}")
        conn = sql.connect(self.raw_dataset_file)

        rsvp_df = self.__get_filtered_rsvp_df(conn=conn)
        event_df, user_df = self.__create_dfs_from_rsvp(rsvp_df,conn=conn)
        del rsvp_df
        gc.collect()
        self.num_users = len(user_df)
        self.num_events = len(event_df)
        event_df = self.__gen_split_mask(event_df)
        event_info_df = self.__get_event_info_df(conn=conn)
        user_info_df = self.__get_user_info_df(conn=conn)
        sample_df = self.__create_sample(event_df)
        user_df = pd.merge(user_df, user_info_df, how="left")
        event_df = pd.merge(event_df, event_info_df, how="left")


        del user_info_df,event_info_df
        gc.collect()
        temp = event_df[["id","lat","hour","weekday","hour","group_category","duration","group_id","topic_ids"]].copy()
        temp["topic_ids"] = temp["topic_ids"].apply(lambda x:"^".join(x))
        temp = temp.rename(columns=dict([(i, "event_"+i) for i in temp.columns]))
        sample_df = pd.merge(sample_df,temp,how="left")
        temp = user_df[["id","group_ids","topic_ids"]].copy()
        temp["group_ids"] = temp["group_ids"].apply(lambda x:"^".join(x))
        temp["topic_ids"] = temp["topic_ids"].apply(lambda x: "^".join(x))
        temp = temp.rename(columns=dict([(i, "user_"+i) for i in temp.columns]))
        sample_df = pd.merge(sample_df, temp, how="left")

        event_df.set_index(["id"], inplace=True)
        user_df.set_index(["id"], inplace=True)

        num_users_has_attend(sample_df,event_df=event_df,user_df=user_df)
        user_has_attend_event(sample_df,event_df=event_df,user_df=user_df)
        user_join_event_group(sample_df, event_df=event_df, user_df=user_df)
        user_event_topic_overlap(sample_df, event_df=event_df, user_df=user_df)
        context_similarity(sample_df,event_df=event_df, user_df=user_df)

        sample_df[sample_df["mask"] == 0].to_csv(os.path.join(self.save_dir, "train.csv"))
        sample_df[sample_df["mask"] == 1].to_csv(os.path.join(self.save_dir, "valid.csv"))
        sample_df[sample_df["mask"] == 2].to_csv(os.path.join(self.save_dir, "test.csv"))
        pass

    def parse_features(self):
        pass

    def __get_filtered_rsvp_df(self, conn):
        logging.info("create base data")
        begin_timestamp = int(self.begin_time.timestamp() * 1000)
        end_timestamp = int(self.end_time.timestamp() * 1000)

        event_df = pd.read_sql(
            "select id from Event where created >= "
            ":begin_timestamp and time< :end_timestamp",
            conn, params={"begin_timestamp": begin_timestamp, "end_timestamp": end_timestamp})
        event_df = event_df.astype(dtype={"id":str}).rename(columns={"id":"event_id"})

        rsvp_df = pd.read_sql(
            "select event_id,user_id,created from Rsvp where response ='yes' and created > :begin_timestamp and created < :end_timestamp",
            conn, params={"begin_timestamp": begin_timestamp, "end_timestamp": end_timestamp})
        rsvp_df = rsvp_df.astype(dtype={"event_id": "str", "user_id": str})
        rsvp_df = pd.merge(event_df,rsvp_df)




        n_records = len(rsvp_df) + 1
        while len(rsvp_df) != n_records:
            n_records = len(rsvp_df)
            temp = rsvp_df.groupby("event_id")["created"].count().reset_index()
            temp = temp[(temp["created"] >= self.event_min_attend) & (temp["created"] <= self.event_max_attend)]
            temp = pd.DataFrame(temp["event_id"].unique(), columns=["event_id"])
            self.events = set(temp["event_id"])
            rsvp_df = rsvp_df.merge(temp)
            temp = rsvp_df.groupby("user_id")["created"].count().reset_index()
            temp = temp[temp["created"] >= self.user_min_attend]
            temp = pd.DataFrame(temp["user_id"].unique(), columns=["user_id"])
            self.users = set(temp["user_id"])
            rsvp_df = rsvp_df.merge(temp)
        convert_df_cols_to_dtype(rsvp_df, ["created"], "datetime", tz_info=self.tz_info, t_scale=1e-3)
        return rsvp_df



    def __create_dfs_from_rsvp(self, rsvp_df, conn):
        logging.info("create rsvp data")
        begin_timestamp = int(self.begin_time.timestamp() * 1000)
        end_timestamp = int(self.end_time.timestamp() * 1000)
        event_info_df = pd.read_sql(
            "select id,created,time from Event where created >= "
            ":begin_timestamp and time< :end_timestamp",
            conn, params={"begin_timestamp": begin_timestamp, "end_timestamp": end_timestamp})

        event_info_df = event_info_df.astype(dtype={"id": str}).rename(columns={"created":"event_created_time","time":"event_begin_time"})
        rsvp_df =  rsvp_df.sort_values(["created", "event_id"])
        event_df = pd.merge(rsvp_df.groupby("event_id")["created"].apply(list).reset_index(), rsvp_df.groupby("event_id")["user_id"].apply(list).reset_index())
        event_df = event_df.rename(columns={"user_id":"attendance_ids", "created":"attendance_times","event_id":"id"})
        event_df = pd.merge(event_df,event_info_df).sort_values("event_created_time")
        convert_df_cols_to_dtype(event_df,["event_created_time","event_begin_time"],"datetime", tz_info=self.tz_info, t_scale=1e-3)


        rsvp_df = rsvp_df.sort_values(["created", "user_id"])
        user_df = pd.merge(rsvp_df.groupby("user_id")["created"].apply(list).reset_index(), rsvp_df.groupby("user_id")["event_id"].apply(list).reset_index())
        user_df = user_df.rename(columns={"event_id":"joined_event_ids", "created":"joined_event_times"})
        self.user2events = dict()
        for u_id,e_ids in zip(user_df["user_id"], user_df["joined_event_ids"]):
            self.user2events[u_id] = set(e_ids)
        user_df.rename(columns={"user_id":"id"},inplace=True)
        return event_df,user_df

    def __gen_split_mask(self, event_df):
        train_pr = self.train_pr
        test_pr = self.test_pr
        val_pr = 1.0 - train_pr - test_pr
        if abs(val_pr) < 1e-4:
            val_pr = 0.0
        mask = np.zeros((self.num_events,), dtype=np.int)
        mask[int(self.num_events * train_pr):int(self.num_events * (val_pr + train_pr))] = 1
        mask[int(self.num_events * (val_pr + train_pr)):] = 2
        event_df["mask"] = mask
        return event_df

    def __create_sample(self, event_df):
        res = []
        tree = self.build_event_query_tree(event_df)
        group_idx = 0
        for e_id, pos_u_ids, attend_times, mask in tqdm.tqdm(zip(event_df["id"], event_df["attendance_ids"], event_df["attendance_times"], event_df["mask"]),total=len(event_df),desc="create samples"):
            temp = list(zip(pos_u_ids[1:], attend_times[1:]))
            sampled_pos = temp[:self.cold_start_reserve_length]
            rest_pos = temp[self.cold_start_reserve_length:]
            shuffle(rest_pos)
            rest_pos = sorted(rest_pos,key=lambda x:x[1])
            sampled_pos += rest_pos[:(self.max_sample_per_event - self.cold_start_reserve_length)]
            for s_id, (pos_u_id, attend_time) in enumerate(sampled_pos):
                neg_events = self._get_valid_events_by_datetime(tree,attend_time)
                if mask == 0 and len(neg_events) - 1 < self.min_neg_per_pos:
                    continue
                shuffle(neg_events)
                neg_events = neg_events[:self.max_neg_per_pos] if mask ==0 else neg_events[:self.eval_max_neg_per_pos]
                res.append([e_id, pos_u_id, attend_time, 1, s_id, group_idx, mask])
                for neg_event in neg_events:
                    if neg_event == e_id or neg_event in self.user2events[pos_u_id]:
                        continue
                    res.append([neg_event, pos_u_id, attend_time, 0, s_id, group_idx,mask])
                group_idx += 1
        res = pd.DataFrame(res, columns=["event_id", "user_id", "rec_time", "label", "sequence_idx", "group_idx","mask"])

        return res
    @staticmethod
    def build_event_query_tree(event_df):
        tree = IntervalTree()
        for c_t, e_t, e_id in zip(event_df["event_created_time"], event_df["event_begin_time"], event_df["id"]):
            tree[c_t:e_t] = e_id
        return tree

    @staticmethod
    def _get_valid_events_by_datetime(tree, t):
        return [item[2] for item in tree.at(t)]

    def __get_event_info_df(self, conn):
        logging.info("create event data")
        event_df = pd.read_sql(
            "select id,created,time,description,group_id,duration,venue_id from Event", conn)
        event_df = event_df.merge(pd.DataFrame(self.events, columns=["id"]))
        event_df = event_df.astype(dtype={"group_id":str})
        convert_df_cols_to_dtype(event_df, "venue_id", "int")
        convert_df_cols_to_dtype(event_df, ["created", "time"], "datetime", tz_info=self.tz_info, t_scale=1e-3)
        convert_df_cols_to_dtype(event_df, "duration", "timedelta", t_scale=1e-3)

        event_df["description"] = event_df["description"].fillna("")
        event_df["description"] = event_df["description"].apply(clean_text)

        tf = TfidfVectorizer(max_features=512)
        event_df["description"] = tf.fit_transform(event_df["description"]).toarray().tolist()

        venue_df = pd.read_sql("select id,lat,lon from Venue", conn)
        convert_df_cols_to_dtype(venue_df, "id", "int")
        convert_df_cols_to_dtype(venue_df, ["lat", "lon"], target_dtype="float")
        venue_df.rename(columns={"id": "venue_id"}, inplace=True)
        event_df = pd.merge(event_df, venue_df, how="left")
        for col in ["lat", "lon"]:
            event_df[col] = event_df[col].fillna(0.0)
            event_df.loc[abs(event_df[col] - 0.0) < 1e-3, col] = np.NaN
            event_df[col] = event_df[col].fillna(event_df[col].mean())
        del venue_df
        gc.collect()

        group_topic_df = pd.read_sql("select group_id,topic_id from GroupTopic", conn)
        convert_df_cols_to_dtype(group_topic_df, ["group_id", "topic_id"])
        group_topic_df = group_topic_df.groupby("group_id")["topic_id"].apply(
            lambda x: list(set(list(x)))).reset_index().rename(columns={"topic_id": "topic_ids"})
        event_df = pd.merge(event_df, group_topic_df, how="left")
        event_df["topic_ids"] = [x if isinstance(x, list) else [] for x in event_df["topic_ids"].values]
        del group_topic_df
        gc.collect()

        group_df = pd.read_sql("select id,category_id from 'Group'", conn)
        convert_df_cols_to_dtype(group_df,"id")
        group_df = group_df.rename(columns={"id":"group_id","category_id":"group_category"})
        event_df = pd.merge(event_df, group_df, how="left")


        event_df["weekday"] = event_df["time"].apply(lambda x: x.weekday())
        event_df["hour"] = event_df["time"].apply(lambda x: x.hour)
        event_df["duration"] = event_df["duration"].apply(lambda x: int(x.total_seconds()))
        event_df["topic_ids"] = event_df["topic_ids"].apply(set)
        event_df.drop(columns=["venue_id",'time','created'], inplace=True)
        return event_df

    def __get_user_info_df(self, conn):
        logging.info("create user data")
        user_df = pd.read_sql(
            "select id,bio,city,country,state,lat,lon,joined from User", conn)
        convert_df_cols_to_dtype(user_df, "id")
        convert_df_cols_to_dtype(user_df, "joined", "datetime", tz_info=self.tz_info, t_scale=1e-3)
        convert_df_cols_to_dtype(user_df, ["lat", "lon"], target_dtype="float")
        user_df = user_df.merge(pd.DataFrame(self.users, columns=["id"]))
        gc.collect()
        user_topic_df = pd.read_sql("select user_id,topic_id from UserTopic", conn)
        convert_df_cols_to_dtype(user_topic_df, ["user_id", "topic_id"])
        user_topic_df = user_topic_df.merge(pd.DataFrame(self.users, columns=["user_id"]))

        user_topic_df = user_topic_df.groupby("user_id")["topic_id"].apply(
            lambda x: list(set(list(x)))).reset_index().rename(columns={"topic_id": "topic_ids", "user_id": "id"})
        user_df = pd.merge(user_df, user_topic_df, how="left")
        user_df["topic_ids"] = [x if isinstance(x, list) else [] for x in user_df["topic_ids"].values]
        del user_topic_df
        gc.collect()

        user_group_df = pd.read_sql("select user_id,group_id from UserGroup", conn)
        convert_df_cols_to_dtype(user_group_df, ["user_id", "group_id"])
        user_group_df = user_group_df.merge(pd.DataFrame(self.users, columns=["user_id"]))
        user_group_df = user_group_df.groupby("user_id")["group_id"].apply(
            lambda x: list(set(list(x)))).reset_index().rename(columns={"group_id": "group_ids", "user_id": "id"})
        user_df = pd.merge(user_df, user_group_df, how="left")
        user_df["group_ids"] = [x if isinstance(x, list) else [] for x in user_df["group_ids"].values]
        del user_group_df
        gc.collect()

        user_df["topic_ids"] = user_df["topic_ids"].apply(set)
        user_df["group_ids"] = user_df["group_ids"].apply(set)
        user_df["joined"] = user_df["joined"].apply(lambda x: x.year)
        return user_df

    def check_can_skip(self):
        files = [os.path.join(self.save_dir, x) for x in ["test.csv","train.csv","valid.csv"]]
        for file in files:
            if not os.path.exists(file):
                return False
        return True
