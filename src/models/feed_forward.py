import data_preparation as dp
import tensorflow as tf
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
import numpy as np
#from models.tokenizer import get_tokenizer
from keras.models import save_model
import os

import pickle

class FeedForward():

    def __init__(self, input_shape= None):
        if input_shape:            
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.Flatten(input_shape=(input_shape,)))
            self.model.add(tf.keras.layers.Dense(256, activation='relu'))
            self.model.add(tf.keras.layers.Dense(15, activation="softmax"))
            self.model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
            return
        raise Exception("Please provide either input_shape or model_path")
    
    def train(self, X_train, y_train, epochs = 10	):
        y = tf.keras.utils.to_categorical(
            y_train.reshape(y_train.shape[0]), num_classes=15, dtype= "int64"
        )
        self.model.fit(X_train, y, epochs=epochs, batch_size=128)

    def save(self, path = "res/models/feed_forward.h5"):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    def predict(self, X_test):
        result = self.model.predict(X_test, verbose=0) 
        result = np.argmax(result, axis=1)
        return result

def load_model(model_path = "res/models/feed_forward.h5"):
      with open(model_path, 'rb') as handle:
        return pickle.load(handle)


