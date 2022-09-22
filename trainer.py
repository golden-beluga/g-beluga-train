from copyreg import pickle
import lightgbm as lgb
import pickle

import dataset

def save_model(model, path):
    file = "./model_data/"+path
    pickle.dump(model, open(file, "wb"))

def train_lambdarank():
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'boosting_type': 'gbdt',
        'num_iterations': 500,
        'max_bin': 100,
        'num_leaves': 50,
        'learning_rate': 0.05,
        'early_stopping_rounds': 50,
    }
    
    dtrain, dval = dataset.load_dataset_for_training_lambdarank()
    model = lgb.train(params, dtrain, valid_sets=dval)
    
    save_model(model, "lambdarank/lgb_model.pkl")
    