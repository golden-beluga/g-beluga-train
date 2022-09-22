import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

import dataset
import utils.read_data as rd
import utils.preprocessing as pp
import utils.join_race_data as jrd
import utils.prepare_data as prepare_data
import utils.io_model as im

import os
from os.path import join, dirname
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(verbose=True)
dotenv_path = join(Path().resolve(), ".env")
load_dotenv(dotenv_path)
GOOGLE_DRIVE_PATH = os.environ.get("GOOGLE_DRIVE_PATH")
TRAIN_DATA_PATH = GOOGLE_DRIVE_PATH + "/train_data"

def predict_using_lambdarank(target_data_path):
    df_for_prediction = dataset.load_df_for_prediction()
    

if __name__ == "__main__":
    target_data_path = GOOGLE_DRIVE_PATH + '/test_data/takarazuka/'
    predict_using_lambdarank(target_data_path)
