import data_preparation as dp
import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tokenizer
class FeedForwardNetwork():
    def __init__(self,input_shape):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(input_shape,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(15, activation="softmax")
        ])
        self.model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        

    def train(self, x_train, y_train, epochs):
        y = tf.keras.utils.to_categorical(
            y_train.to_numpy().reshape(y_train.shape[0]), num_classes=15, dtype= "int64"
        )
        x = tokenizer.get_tokenized(x_train)
        self.model.fit(x, y, epochs=epochs, batch_size=128)

    def predict(self, x_test):
        x_test = tokenizer.get_tokenized(x_test)
        result = self.model.predict(x_test) 
        result = np.argmax(result, axis=1)
        return result