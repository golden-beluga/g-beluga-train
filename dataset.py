import os
from os.path import join
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import lightgbm as lgb
import pandas as pd

import utils.read_data as rd
import utils.io_model as io_m
import utils.preprocessing as pp
import utils.prepare_data as prepare_data
import utils.join_race_data as join_race_data

load_dotenv(verbose=True)
dotenv_path = join(Path().resolve(), '.env')
load_dotenv(dotenv_path)
GOOGLE_DRIVE_PATH = os.environ.get("GOOGLE_DRIVE_PATH")
DATA_PATH = GOOGLE_DRIVE_PATH + '/train_data'

def make_label(rank):
    rank = str(rank)
    if not(rank.isdigit()):
        rank = 30

    return int(rank)

def read_dataset_csv():
    df_dataset = rd.read_horse_race_csv(DATA_PATH)
    df_dataset = df_dataset.sort_values("race_id", ascending=True)
    return df_dataset

def make_query(df_dataset):
    return list(df_dataset.groupby('race_id').count().race_course)

def rank_to_label(df_dataset, training=True):
    if training:
        df_dataset["label"] = df_dataset["rank"].apply(make_label)
    df_dataset["rank-1"] = df_dataset["rank-1"].apply(make_label)
    df_dataset["rank-2"] = df_dataset["rank-2"].apply(make_label)
    df_dataset["rank-3"] = df_dataset["rank-3"].apply(make_label)
    
    return df_dataset

def load_dataset_for_training_lambdarank():
    df_dataset = read_dataset_csv()
    query = make_query(df_dataset)
    df_dataset = rank_to_label(df_dataset)
    df_for_learning = prepare_data.prepare_train_data(df_dataset)
    columns_for_learning = df_for_learning.columns.values.tolist()
    columns_for_learning.remove("label")
    
    # データセットを入力データとラベルデータに分ける
    x = np.array(df_for_learning[columns_for_learning])
    y = np.array(df_for_learning['label'])
    
    # trainとtestデータに分ける
    split = int(len(query) / 5)
    query_train = query[:split]  
    x_train = x[:sum(query[:split])]
    y_train = y[:sum(query[:split])]
    query_test = query[split:] 
    x_test = x[sum(query[:split]):]
    y_test = y[sum(query[:split]):]
    
    dtrain = lgb.Dataset(x_train, y_train, group=query_train)
    dval = lgb.Dataset(x_test, y_test, reference=dtrain, group=query_test)
    
    return dtrain, dval

def add_past_race_data(target_df, past_df, n):
    # ターゲットデータに過去nレース分の情報を追加
    columns_past_data = [c for c in past_df.columns if "-" not in c]
    target_df_added_past_data = join_race_data.join_n_race_for_test_data(
        past_df[columns_past_data], 
        target_df, 
        n
    )
    
    return target_df_added_past_data

def load_df_for_prediction(target_data_path):
    target_horse_df = rd.read_target_horse_csv(target_data_path)
    target_race_df  = rd.read_target_race_csv(target_data_path)
    target_df = pd.merge(target_horse_df, target_race_df, on='race_id', how='left')
    
    past_df = rd.read_horse_race_csv(DATA_PATH)
    df_for_prediction = add_past_race_data(target_df, past_df, 3)
    
    return df_for_prediction, past_df

def load_df_for_prediction_of_lambdarank(target_data_path):
    df_for_prediction, past_df = load_df_for_prediction(target_data_path)
    
    df_for_prediction = rank_to_label(df_for_prediction, False)
    past_df = rank_to_label(past_df, False)
    
    df_for_prediction = prepare_data.prepare_data_for_prediction(
        df_for_prediction, 
        past_df, 
        use_default_make_label=True
    )
    
    return df_for_prediction
