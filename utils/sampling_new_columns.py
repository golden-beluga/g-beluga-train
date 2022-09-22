import pandas as pd
import tensorflow as tf
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import re

def sampling_class(race_class):
    """
    レースの出場条件を統一して取得(2019年から出場条件の呼び方が変わったため新しい呼び方に統一)

    Parameters
    ------------
    race_class : object
        レースの出場条件

    Returns
    ------------
    sample_class : object
        レースの出場条件(勝利数)
    """
    if type(race_class) != str:
        return race_class

    if "新馬" in race_class:
        sample_class="新馬"
    elif "未勝利" in race_class:
        sample_class="未勝利"
    elif "1勝クラス" in race_class or "500万下" in race_class:
        sample_class="1勝クラス"
    elif "2勝クラス" in race_class or "1000万下" in race_class:
        sample_class="2勝クラス"
    elif "3勝クラス" in race_class or "1600万下" in race_class:
        sample_class="3勝クラス"
    else :
        sample_class="オープン"
    return sample_class

def sampling_grade(race_title,race_class):
    """
    レースのランクを取得
    (G1>G2>G3>G>オープン>3勝クラス>2勝クラス>1勝クラス>新馬≥未勝利: jがついてるのは障害レース)

    Parameters
    ------------
    race_title : object
        レースの名前
    race_class : object
        レースの出場条件

    Returns
    ------------
    race_grade : object
        レースランク
    """
    if type(race_title) != str:
        return race_title

    if "G" in race_title and "T" not in race_title:
        race_grade=re.split("[()]",race_title)[-2]
    else :
        race_grade=sampling_class(race_class)
    return race_grade

def sampling_distance(race_course):
    """
    コースの距離を取得

    Parameters
    ------------
    race_course : object
        コースの情報

    Returns
    ------------
    race_distance : object
        コースの距離
    """
    if type(race_course) != str:
        return race_course
    
    race_distance=re.sub(r"\D", "",race_course)
    return race_distance

def sampling_direction(race_course):
    """
    コースの向きを取得
    障害レースかどうか判定

    Parameters
    ------------
    race_course : object
        コースの情報

    Returns
    ------------
    race_direction : object
        コースの向き
    """
    if type(race_course) != str:
        return race_course

    if "障" in race_course:
        race_direction="障"
    elif "右" in race_course:
        race_direction="右"
    elif "左" in race_course:
        race_direction="左"
    else:
        race_direction="直"
    return race_direction

def sampling_race_genaration(race_class):
    """
    レースの出場世代を取得(世:同じ年齢の馬だけが出る世代限定戦 ,古:古馬混合戦)

    Parameters
    ------------
    race_class : object
        コースの情報

    Returns
    ------------
    race_genaration : object
        レースの出場条件(世代)
    """
    if type(race_class) != str:
        return race_class
    
    if "以上" in race_class:
        race_genaration="世"
    else:
        race_genaration="古"
    return race_genaration

def kyakusitu_code_c(horse_number,half_way_rank):
    # 欠損値の場合
    if type(half_way_rank) != str:
        return half_way_rank
     
    tuuka= [int(s) for s in half_way_rank.split('-')]
    if len(tuuka)==1:
        if tuuka[0]==1:
            kyakusitu_code=0
        elif tuuka[0]>=2 and tuuka[0]<=4:
            kyakusitu_code=1
        elif tuuka[0]<=(horse_number/3*2) and horse_number>=8:
            kyakusitu_code=2
        else:
            kyakusitu_code=3
    if len(tuuka)>=2:
        last_couner=tuuka.pop()
        if 1 in tuuka:
            kyakusitu_code=0
        elif last_couner>=2 and last_couner<=4:
            kyakusitu_code=1
        elif last_couner<=(horse_number/3*2) and horse_number>=8:
            kyakusitu_code=2
        else:
            kyakusitu_code=3
    return kyakusitu_code

def extract_place(where_racecourse):
    place = re.match(r"\d+回(\D+)\d+日目", where_racecourse).groups()[0]
    return place

def to_seconds(time):
    if type(time) != str:
        return time
    minutes, second_milliseconds = time.split(":")
    seconds, milliseconds = second_milliseconds.split(".")
    td = datetime.timedelta(minutes=int(minutes), seconds=int(seconds), milliseconds=int(milliseconds))
    return td.total_seconds()

def extract_weight(horse_weight):
    if horse_weight == "計不" or type(horse_weight) != str:
        return -1
    weight = re.match(r"(\d+)\([+-]?\d+\)", horse_weight).groups()[0]
    return weight 