import pandas as pd
import os
import sys


def read_horse_csv(path):
    df = pd.DataFrame()
    for file in os.listdir(path):
        base, ext = os.path.splitext(file)
        if ext == '.csv' and 'horse' in base:
            df = pd.concat([df, pd.read_csv(path + '/' + file)])
    df = df.astype({'race_id': str})
    df = df.query('not race_id.str.contains("\(", na=False)', engine='python')
    df = df.astype({'race_id': float, 'horse_id': int, 'popular': 'float16', 'burden_weight': 'float16', 'frame_number': 'int8', 'horse_number': 'int8', 'total_horse_number': 'int8', 'rider_id': 'int32', 'tamer_id': 'int32', 'last_time': 'float16'})

    return df.reset_index(drop=True)


def read_race_csv(path):
    df = pd.DataFrame()
    for file in os.listdir(path):
        base, ext = os.path.splitext(file)
        if ext == '.csv' and 'race' in base:
            df = pd.concat([df, pd.read_csv(path + '/' + file)])
    df = df.astype({'race_id': str})
    df = df.query('not race_id.str.contains("\(", na=False)', engine='python')
    df = df.astype({'race_id': float, 'total_horse_number': 'int8', 'frame_number_first': 'int8', 'horse_number_first': 'int8', 'frame_number_second': 'int8', 'horse_number_second': 'int8', 'frame_number_third': 'int8', 'horse_number_third': 'int8'})
    
    return df.reset_index(drop=True)


def read_dir_horse_csv(path):
    df_horse = pd.DataFrame()
    for f in os.listdir(path):
         if os.path.isdir(os.path.join(path, f)):
            print(f)
            df_horse = df_horse.append(read_horse_csv(path + '/' + f + '/'))
            
    return df_horse.reset_index(drop=True)


def read_dir_race_csv(path):
    df_race = pd.DataFrame()
    for f in os.listdir(path):
         if os.path.isdir(os.path.join(path, f)):
            print(f)
            df_race = df_race.append(read_race_csv(path + '/' + f + '/'))
            
    return df_race.reset_index(drop=True)
    


def read_horse_race_csv(path):
    df = pd.DataFrame()
    for file in os.listdir(path):
        base, ext = os.path.splitext(file)
        if ext == '.csv' and 'horse_race' in base:
            print(file)
            df = df.append(pd.read_csv(path + '/' + file, index_col=0))
    df = df.astype({'race_id': float})

    return df.reset_index(drop=True)


def read_target_horse_csv(path):
    df = pd.DataFrame()
    for file in os.listdir(path):
        base, ext = os.path.splitext(file)
        if ext == '.csv' and 'horse' in base:
            df = pd.concat([df, pd.read_csv(path + '/' + file)])
    df = df.astype({'race_id': float, 'horse_id': int})
    
    return df.reset_index(drop=True)

def read_target_race_csv(path):
    df = pd.DataFrame()
    for file in os.listdir(path):
        base, ext = os.path.splitext(file)
        if ext == '.csv' and 'race' in base:
            df = pd.concat([df, pd.read_csv(path + '/'  + file)])
    df = df.astype({'race_id': float})
  
    return df.reset_index(drop=True)
