import re
import datetime
import pandas as pd
import tensorflow as tf
import numpy as np
#from tqdm import tqdm

def fill_missing_columns(df_a, df_b):
    columns_for_b = set(df_a.columns) - set(df_b.columns)
    for column in columns_for_b:
        df_b[column] = 0

    columns_for_a = set(df_b.columns) - set(df_a.columns)
    for column in columns_for_a:
        df_a[column] = 0


def one_hot_encoding(df):
    return pd.get_dummies(df)

def make_label(rank_values, horse_number_values):
    labels = []
    high = 1 / 3
    mid = 2 / 3
    for rank, horse_number in zip(rank_values, horse_number_values):
        # 欠損値の場合
        if rank == -1:
            labels.append(rank)
            continue
        # 順位が付かないデータに関しては最低レベルのラベルを付与
        not_rank = False
        # "(" => ...(再)　or ...(降)
        for c in ["中", "取", "除", "(", "失"]:
            if c in str(rank):
                labels.append("low")
                not_rank = True
                break
        if not_rank:
            continue
            
        relative_rank = int(rank) / horse_number
    
        if relative_rank < high:
            labels.append("high")
        elif relative_rank < mid:
            labels.append("middle")
        else:
            labels.append("low")
            
    return labels
