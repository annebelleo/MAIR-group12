import data_preparation as dp
import tensorflow as tf
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tokenizer
from keras.models import save_model
import os

model_path = "res/models/feed_forward.h5"
class FeedForwardNetwork(tf.keras.Sequential):
    def __init__(self,input_shape):
        super().__init__()
        self.add(tf.keras.layers.Flatten(input_shape=(input_shape,)))
        self.add(    tf.keras.layers.Dense(256, activation='relu'))
        self.add(tf.keras.layers.Dense(128, activation='relu'))
        self.add(tf.keras.layers.Dense(15, activation="softmax"))
        self.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        

    def train(self, x_train, y_train, epochs):
        y = tf.keras.utils.to_categorical(
            y_train.to_numpy().reshape(y_train.shape[0]), num_classes=15, dtype= "int64"
        )
        x = tokenizer.get_tokenized(x_train)
        self.fit(x, y, epochs=epochs, batch_size=128)

    def predict(self, x_test):
        x_test = tokenizer.get_tokenized(x_test)
        result = super().predict(x_test) 
        result = np.argmax(result, axis=1)
        return result
    
def get_trained_model():
    if os.path.exists(model_path):
        model =FeedForwardNetwork(input_shape=tokenizer.get_size())
        model.load_weights(model_path) 
        return model

    dialogDF = dp.get_data(drop_duplicates=False)
    dialogTrain, dialogTest = train_test_split(dialogDF, test_size=0.15, random_state=42)
    model = FeedForwardNetwork(input_shape=tokenizer.get_size())
    model.train(dialogTrain["sentence"], dialogTrain["label"], epochs=2)
    model.save_weights(model_path)
    return model