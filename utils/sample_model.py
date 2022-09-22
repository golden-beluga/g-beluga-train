import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

class HorseModel(Model):
    def __init__(self, i_dimention, o_dimention):
        super(HorseModel, self).__init__()
        self.d1 = Dense(i_dimention, activation='relu')
        self.d2 = Dense(32, activation='relu')
        self.d3 = Dense(128, activation='relu')
        self.d4 = Dense(o_dimention, activation='softmax')
        
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.d4(x)
    