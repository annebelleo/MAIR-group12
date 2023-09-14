import data_preparation as dp
import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
class FeedForwardNetwork():
    def __init__(self,x, y, epochs=10):
        # categories y:
        y = tf.keras.utils.to_categorical(
            y.to_numpy().reshape(y.shape[0]), num_classes=15, dtype= "int64"
        )
        # create the tokenizer for x:
        self.tokenizer = Tokenizer()
        # fit the tokenizer on the documents
        self.tokenizer.fit_on_texts(x)
        # integer encode documents
        x = self.tokenizer.texts_to_matrix(x, mode='count')
        # split in test and training data:
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(x.shape[1],)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(15, activation="softmax")
        ])
        self.model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        
        self.model.fit(x, y, epochs=epochs, batch_size=128)
        

    def predict(self, x_test):
        x_test = self.tokenizer.texts_to_matrix(x_test , mode='count')
        result = self.model.predict(x_test) 
        result = np.argmax(result, axis=1)
        return result