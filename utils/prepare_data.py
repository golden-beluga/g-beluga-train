import pandas as pd
import numpy as np
import utils.preprocessing as pp
import utils.sampling_new_columns as sc

columns_after_processing = ["weather", 
                 "race_grade", "race_generation", "race_distance",
                 "race_direction", "ground_status",
                 "frame_number", "horse_number",
                 "sex", "age", "burden_weight", "rider_id", 
                 "tamer_id", "horse_weight",
                 "rank-1", "rank-2", "rank-3", 
                 "goal_time-1", "goal_time-2", "goal_time-3",
                 "last_time-1", "last_time-2", "last_time-3", 
                 "kyakusitu-1", "kyakusitu-2", "kyakusitu-3", 
                 "odds-1", "odds-2", "odds-3",
                 "popular-1", "popular-2", "popular-3", 
                 "prize-1", "prize-2", "prize-3"]

def prepare_train_data(raw_df: pd.DataFrame, use_default_make_label: bool = False, one_hot: bool = True) -> pd.DataFrame:
    """
    生データを学習用のデータに加工する
    
    Parameters
    ----------
    raw_df : pandas.DataFrame
        生データのデータフレーム
        
    use_default_make_label : bool = False
        デフォルトのmake_labelを使用するか
    
    one_hot : bool = False
        one-hotエンコーディングを行うか
    
    Returns
    -------
    df_for_learning : pandas.DataFrame
        学習用データフレーム
    """
    df_for_learning = process_features(raw_df)
    # ラベル作成
    if use_default_make_label:
        df_for_learning.loc[: ,"label"] = pp.make_label(df_for_learning.loc[: ,"rank"].values, df_for_learning.loc[: ,"total_horse_number_x"].values)
        df_for_learning.loc[: ,"rank-1"] = pp.make_label(df_for_learning.loc[: ,"rank-1"].values, df_for_learning.loc[: ,"total_horse_number_x-1"].values)
        df_for_learning.loc[: ,"rank-2"] = pp.make_label(df_for_learning.loc[: ,"rank-2"].values, df_for_learning.loc[: ,"total_horse_number_x-2"].values)
        df_for_learning.loc[: ,"rank-3"] = pp.make_label(df_for_learning.loc[: ,"rank-3"].values, df_for_learning.loc[: ,"total_horse_number_x-3"].values)
    # one-hotベクトル化
    df_for_learning = df_for_learning[columns_after_processing+["label"]]
    if one_hot:
        df_for_learning = pp.one_hot_encoding(df_for_learning)

    return df_for_learning
    
def prepare_data_for_prediction(df_for_prediction: pd.DataFrame, past_data_df: pd.DataFrame, use_default_make_label: bool = False, one_hot: bool = True) -> pd.DataFrame:
    """
    データを予測に使えるように加工
    
    Parameters
    ----------
    df_for_prediction : pandas.DataFrame
        予測に使うデータ
        
    past_data_df : pandas.DataFrame
        学習に使った過去データ、予測用データには一部のカテゴリデータしかないので、
        one-hotベクトル化後、学習済みモデルの入力次元数と次元数が合わない。
        そこで、過去データを参照して追加する。
    
    use_default_make_label : bool = False
        デフォルトのmake_labelを使用するか
        (現状はサンプルの多層NNで使う)
    
    one_hot : bool = True
        one-hotエンコーディングを行うか
    
    Retures
    -------
    df_after_preprocessing : pandas.DataFrame
        加工済み予測用データ
    """
    df_after_preprocessing = process_features(df_for_prediction)
    past_data_df_after_preprocessing = process_features(past_data_df)
    
    if use_default_make_label:
        df_after_preprocessing.loc[: ,"rank-1"] = pp.make_label(df_after_preprocessing.loc[: ,"rank-1"].values, df_after_preprocessing.loc[: ,"total_horse_number_x-1"].values)
        df_after_preprocessing.loc[: ,"rank-2"] = pp.make_label(df_after_preprocessing.loc[: ,"rank-2"].values, df_after_preprocessing.loc[: ,"total_horse_number_x-2"].values)
        df_after_preprocessing.loc[: ,"rank-3"] = pp.make_label(df_after_preprocessing.loc[: ,"rank-3"].values, df_after_preprocessing.loc[: ,"total_horse_number_x-3"].values)
        past_data_df_after_preprocessing.loc[: ,"rank-1"] = pp.make_label(past_data_df_after_preprocessing.loc[: ,"rank-1"].values, past_data_df_after_preprocessing.loc[: ,"total_horse_number_x-1"].values)
        past_data_df_after_preprocessing.loc[: ,"rank-2"] = pp.make_label(past_data_df_after_preprocessing.loc[: ,"rank-2"].values, past_data_df_after_preprocessing.loc[: ,"total_horse_number_x-2"].values)
        past_data_df_after_preprocessing.loc[: ,"rank-3"] = pp.make_label(past_data_df_after_preprocessing.loc[: ,"rank-3"].values, past_data_df_after_preprocessing.loc[: ,"total_horse_number_x-3"].values)
    
    df_after_preprocessing = df_after_preprocessing[columns_after_processing]
    past_data_df_after_preprocessing = past_data_df_after_preprocessing[columns_after_processing]
    if one_hot:
        df_after_preprocessing = pp.one_hot_encoding(df_after_preprocessing)
        past_data_df_after_preprocessing = pp.one_hot_encoding(past_data_df_after_preprocessing)
    
    pp.fill_missing_columns(df_after_preprocessing, past_data_df_after_preprocessing)
    
    return df_after_preprocessing
    
def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    特徴量を加工
    ※ DataFrameは参照渡しなので、引数で渡したデータへの変更は
       関数での処理後も元データに残る
        
    Parameters
    ----------
    df : pandas.DataFrame
        特徴量を加工するデータフレーム
    
    Returns
    -------
    df_after_processing: pandas.DataFrame
        特徴量加工後のデータフレーム
    """
    df_after_processing = df
    
    df_after_processing.loc[:, "race_class"] = df_after_processing.loc[:, "race_class"].map(sc.sampling_class)    
    df_after_processing.loc[:, "race_grade"] = df_after_processing.apply(lambda row: sc.sampling_grade(row["race_title"], row["race_class"]), axis=1)
    df_after_processing.loc[:, "race_generation"] = df_after_processing.loc[:, "race_class"].map(sc.sampling_race_genaration)
    df_after_processing.loc[:, "race_distance"] = df_after_processing.loc[:, "race_course"].map(sc.sampling_distance).astype(np.int16)
    df_after_processing.loc[:, "race_direction"] = df_after_processing.loc[:, "race_course"].map(sc.sampling_direction)

    df_after_processing.loc[: ,"sex"] = df_after_processing.loc[: ,"sex_and_age"].map(lambda sex_and_age: sex_and_age[0])
    df_after_processing.loc[: ,"age"] = df_after_processing.loc[: ,"sex_and_age"].map(lambda sex_and_age: sex_and_age[1:]).astype(np.int16)
    
    df_after_processing.loc[: ,"goal_time-1"] = df_after_processing.loc[: ,"goal_time-1"].map(sc.to_seconds)
    df_after_processing.loc[: ,"goal_time-2"] = df_after_processing.loc[: ,"goal_time-2"].map(sc.to_seconds)
    df_after_processing.loc[: ,"goal_time-3"] = df_after_processing.loc[: ,"goal_time-3"].map(sc.to_seconds)
    
    df_after_processing.loc[: ,"horse_weight"] = df_after_processing.loc[: ,"horse_weight"].map(sc.extract_weight).astype(np.int64)
    
    df_after_processing.loc[: ,"prize-1"] = df_after_processing.loc[: ,"prize-1"].map(
        lambda prize: prize.replace(",", "") if type(prize) == str else prize).astype(np.float32)
    df_after_processing.loc[: ,"prize-2"] = df_after_processing.loc[: ,"prize-2"].map(
        lambda prize: prize.replace(",", "") if type(prize) == str else prize).astype(np.float32)
    df_after_processing.loc[: ,"prize-3"] = df_after_processing.loc[: ,"prize-3"].map(
        lambda prize: prize.replace(",", "") if type(prize) == str else prize).astype(np.float32)
    
    df_after_processing.loc[: ,"kyakusitu-1"] = [sc.kyakusitu_code_c(n, r) 
        for n, r in zip(df_after_processing.loc[: ,"total_horse_number_x-1"].values, df_after_processing.loc[: ,"half_way_rank-1"])]
    df_after_processing.loc[: ,"kyakusitu-2"] = [sc.kyakusitu_code_c(n, r) 
        for n, r in zip(df_after_processing.loc[: ,"total_horse_number_x-2"].values, df_after_processing.loc[: ,"half_way_rank-2"])]
    df_after_processing.loc[: ,"kyakusitu-3"] = [sc.kyakusitu_code_c(n, r) 
        for n, r in zip(df_after_processing.loc[: ,"total_horse_number_x-3"].values, df_after_processing.loc[: ,"half_way_rank-3"])]
    
    # 欠損値処理
    df_after_processing = df_after_processing.replace('---', -1)
    df_after_processing = df_after_processing.fillna(-1)

    df_after_processing.loc[: ,"odds-1"] = df_after_processing.loc[: ,"odds-1"].astype(np.float32)
    df_after_processing.loc[: ,"odds-2"] = df_after_processing.loc[: ,"odds-2"].astype(np.float32)
    df_after_processing.loc[: ,"odds-3"] = df_after_processing.loc[: ,"odds-3"].astype(np.float32)
    
    return df_after_processing
