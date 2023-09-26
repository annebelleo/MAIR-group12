import data_preparation as dp
import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from models.tokenizer import get_tokenizer

class DecisionTree():

    def __init__(self, max_depth = None):
        self.model = tree.DecisionTreeClassifier( max_depth=max_depth)
        
    def train(self, x, y):
        # categories y:
        y = y.to_numpy().reshape(y.shape[0])
        tokenizer =  get_tokenizer()

        x = tokenizer.texts_to_matrix(x, mode='count')
        self.model.fit(x, y)
    

    def predict(self, x):
        tokenizer =  get_tokenizer()
        x = tokenizer.texts_to_matrix(x, mode='count')
        result = self.model.predict(x) 
        return result