import data_preparation as dp
import tensorflow as tf
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
import numpy as np
#from models.tokenizer import get_tokenizer
from keras.models import save_model
import os


model_default_path = "res/models/feed_forward.h5"

def get_model(input_shape= None, model_path = model_default_path, overwrite : bool = False):
    if not overwrite:
        if os.path.exists(model_path):
            return keras.models.load_model(model_path) 
    if not input_shape:
        raise Exception("Input shape requert")
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(input_shape,)))
    model.add(    tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(15, activation="softmax"))
    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    model.save(model_path)
    return model

def train(model, X_train, y_train, epochs, save= True, path_save = model_default_path):
    y = tf.keras.utils.to_categorical(
        y_train.reshape(y_train.shape[0]), num_classes=15, dtype= "int64"
    )
    #tokenizer = get_tokenizer()
    #X_train = tokenizer.texts_to_matrix(X_train, mode='count')
    model.fit(X_train, y, epochs=epochs, batch_size=128)
    if save:
        with open(path_save, 'wb') as handle:
            model.save(path_save)

def predict(model, X_test):
    if type(X_test) == str:
        X_test = pd.Series(X_test)
    #tokenizer = get_tokenizer()
    #x_test = tokenizer.texts_to_matrix(x_test, mode='count')
    
    result = model.predict(X_test, verbose=0) 
    result = np.argmax(result, axis=1)
    return result


    
