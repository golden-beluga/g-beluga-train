import pandas as pd
import numpy as np
from typing import Tuple

PRICE_OF_BETTING_TICKET = 100

def extract_race(horse_race_df, race_id):
    return horse_race_df[horse_race_df["race_id"] == race_id]

# n: n-box を引数にとり任意のboxに対応させる(top-5まで、5より大きい場合は警告を出す)
def tansyo_ret(target_ranks: np.array, predicted_ranks: np.array, tansyo_prize: np.array) -> Tuple[int, float]:
    if target_ranks.dtype.type != np.int64:
        error_message = f"target_ranks.dtype is {target_ranks.dtype}, but target_ranks.dtype must be np.int32"
        print(error_message)
        raise error_message
    
    index_of_top_horse = target_ranks == 1
    num_hit = (predicted_ranks[index_of_top_horse] == 1).sum()
    
    if num_hit > 0:
        ret = int(tansyo_prize[index_of_top_horse]) - PRICE_OF_BETTING_TICKET
    else:
        ret = 0
    
    return num_hit, ret

def hukusyo_ret(target_ranks: np.array, predicted_ranks: np.array, hukusyo_prize: np.array) -> Tuple[int, float]:
    if target_ranks.dtype.type != np.int64:
        error_message = f"target_ranks.dtype is {target_ranks.dtype}, but target_ranks.dtype must be np.int32"
        print(error_message)
        raise error_message
    
    index_of_predicted_top_horse = predicted_ranks == 1
    if len(target_ranks) > 7:
        num_hit = (target_ranks[index_of_predicted_top_horse] <= 3).sum()
    else:
        num_hit = (target_ranks[index_of_predicted_top_horse] <= 2).sum()
    
    if num_hit > 0:
        ret = int(hukusyo_prize[index_of_predicted_top_horse]) - PRICE_OF_BETTING_TICKET
    else:
        ret = 0
    
    return num_hit, ret

def top_1_box(race_horse_df: pd.DataFrame, predicted_ranks_df: pd.DataFrame):
    stats = {"tansyo": {"correct_rate": 0, "ret_rate": 0, "ret_std": 0, "hit_list": [], "ret_list": [], "total_hit": 0, "total_ret": 0}}

    race_ids = list(set(race_horse_df["race_id"].values))
    for race_id in race_ids:
        predicted_ranks = extract_race(predicted_ranks_df, race_id)["rank"].values
        target_race_df = extract_race(race_horse_df, race_id)
        is_hit, ret = tansyo_ret(target_race_df["rank"].values, predicted_ranks, target_race_df["tansyo"].values)
        stats["tansyo"]["hit_list"].append(int(is_hit))
        stats["tansyo"]["ret_list"].append(ret)
    
    return stats
