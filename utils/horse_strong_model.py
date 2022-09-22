import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

class HorseStrongModel(Model):
    def __init__(self, i_dimention, o_dimention):
        super(HorseStrongModel, self).__init__()
        self.input_layer = Dense(i_dimention, activation='relu')
        self.latent_output_layer = Dense(32, activation='relu') # 潜在表現の出力にも使うレイヤー
        self.d3 = Dense(32, activation='relu')
        self.d4 = Dense(o_dimention, activation='softmax')

    def call(self, x):
        x = self.input_layer(x)
        x = self.latent_output_layer(x)
        x = self.d3(x)
        return self.d4(x)
    