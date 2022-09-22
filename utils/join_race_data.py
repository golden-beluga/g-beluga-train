import pandas as pd
import numpy as np


def join_row(df, df_row_target, n_race):
    df = df.iloc[0:n_race, :]
    for i in range(1, n_race+1):
        df_row_i = df.iloc[i-1:i, :].reset_index(drop=True)
        df_row_i.columns = list(map(lambda c: c+f"-{i}", df.columns))
        df_row_target = pd.concat([df_row_target, df_row_i], axis=1)
    return df_row_target


def join_n_race_for_train_data(df, arr_race_id, n_race=5):
    res = pd.DataFrame()
    for race_id in arr_race_id:
        df_tmp = df.query('race_id <= @race_id').sort_values("race_id", ascending=False).reset_index(drop=True)
        df_row_target = df_tmp.iloc[0:1]
        df_tmp = df_tmp.drop(0).reset_index(drop=True)
        res = res.append(join_row(df_tmp, df_row_target, n_race))

    return res.reset_index(drop=True)


def join_n_race_for_test_data(df, df_target, n_race=3):
    res = pd.DataFrame()
    arr_horse_id = list(df_target['horse_id'])
    for horse_id in arr_horse_id:
        df_tmp = df.query('horse_id == @horse_id').sort_values("race_id", ascending=False).reset_index(drop=True)
        df_row_target = df_target.query('horse_id == @horse_id').reset_index(drop=True)
        res = res.append(join_row(df_tmp, df_row_target, n_race))

    return res.reset_index(drop=True)


# 古いコード
def add_n_race_data(df_race, df_horse, n_race=1):
    res = pd.DataFrame()
    horse_ids = list(set(df_horse['horse_id']))
    df = pd.merge(df_horse, df_race, on='race_id', how='left')
    for horse_id in tqdm(horse_ids):
        df_horse_tmp = df.query('horse_id == @horse_id')
        df_horse_tmp = df_horse_tmp.sort_values("race_id")
        race_ids = list(set(df_horse_tmp['race_id']))
        for race_id in race_ids:
            df_race_tmp = df_horse_tmp.query('race_id == @race_id').reset_index(drop=True)
            df_horse_low_race_id = df_horse_tmp.query('race_id <= @race_id')
            columns = df_horse_low_race_id.columns
            for n in range(1, n_race+1):
                add_columns = list(map(lambda c: c+f"-{n}", columns))
                add_race = df_horse_low_race_id.iloc[-1-n:-n, :].reset_index(drop=True)
                add_race.columns = add_columns
                df_race_tmp = pd.concat([df_race_tmp, add_race], axis=1)
                    
            res = pd.concat([res, df_race_tmp])

    return res.reset_index(drop=True)


def once_join_n_race_para(df_horse):
    res = pd.DataFrame()
    n_race = 15
    arr_race_id = list(set(df_horse['race_id']))
    for race_id in arr_race_id:
        df_row_target = df_horse.query('race_id == @race_id').reset_index(drop=True)
        df_horse = df_horse.query('race_id < @race_id').sort_values("race_id", ascending=False).reset_index(drop=True)
        columns = df_horse.columns
        for n in range(1, n_race+1):
            df_row_n_race = df_horse.iloc[n-1:n, :].reset_index(drop=True)
            df_row_n_race.columns = list(map(lambda c: c+f"-{n}", columns))
            df_row_target = pd.concat([df_row_target, df_row_n_race], axis=1)

        res = res.append(df_row_target)

    return res.reset_index(drop=True)

