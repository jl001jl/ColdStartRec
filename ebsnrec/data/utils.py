import logging
import os
from datetime import datetime, timezone, timedelta
from typing import List, Union

import h5py
from bs4 import BeautifulSoup
import pandas as pd
# import torch
# from PIL import Image
# import clip


def convert_df_cols_to_dtype(df: pd.DataFrame, cols: Union[List[str], str], target_dtype: str="str", fill_na: bool = True,
                             fill_na_value=0, **kwargs) -> None:
    if not isinstance(cols, list):
        cols = [cols]
    for col in cols:
        if fill_na:
            df[col] = df[col].fillna(fill_na_value)
        if target_dtype == "str":
            df[col] = df[col].astype(str)
        elif target_dtype == "float":
            df[col] = df[col].astype(float)
        elif target_dtype == "int":
            df[col] = df[col].astype(int)
        elif target_dtype == "datetime":
            tz_info = kwargs.get("tz_info", timezone(timedelta(hours=0)))
            t_scale = kwargs.get("t_scale", 1 / 1000)
            df[col] = df[col].apply(lambda x: datetime.fromtimestamp(int(x * t_scale), tz=tz_info))
        elif target_dtype == "timedelta":
            t_scale = kwargs.get("t_scale", 1 / 1000)
            df[col] = df[col].apply(lambda x: timedelta(seconds=int(x * t_scale)))


def binary_search(nums, target):
    l,h = -1,len(nums)- 1
    while l<h:
        mid = (l+h+1)//2
        if nums[mid] >= target:
            h = mid - 1
        else:
            l = mid
    return l


def clean_text(str):
    soup = BeautifulSoup(str, parser="lxml",features="lxml")
    return soup.get_text()