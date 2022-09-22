import tensorflow as tf
from tensorflow import keras

MODEL_DATA_PATH = "model_data/"

def read_model(model_name):
    model = tf.keras.models.load_model(MODEL_DATA_PATH + model_name)
    print(f"{model_name} is read.")
    return model

def save_model(model, model_name="sample_model"):
    model.save(MODEL_DATA_PATH + model_name)
    print(f"{model_name} is saved.")
    
def test():
    import os
    os.chdir("../")
    import numpy as np
    import sample_model
    
    # テスト用の学習済みモデルを作成
    model = sample_model.HorseModel(5)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    train_data = np.random.rand(100, 5)
    train_target = np.array([1 for _ in range(50)] + [0 for _ in range(50)])
    model.fit(train_data, train_target, epochs=5)
    
    file_name = "test_model"
    save_model(model, file_name)
    if os.path.exists(MODEL_DATA_PATH + file_name):
        print("save_model: ok")
    else:
        print(f"save_model: error (not found {file_name})")
    
    new_model = read_model(file_name)
    loss, acc = new_model.evaluate(train_data, train_target, verbose=2)
    if new_model.predict(train_data).shape[1] == 2: 
        print("read_model: ok")
    else:
        print(f"read_model: error (output must be 2 dimensions.)")
    
if __name__ == "__main__":
    test()