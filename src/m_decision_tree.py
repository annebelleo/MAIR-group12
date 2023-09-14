import data_preparation as dp
import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree


class DecisionTree():

    def __init__(self, x, y):
        # categories y:
        y = y.to_numpy().reshape(y.shape[0])
        # create the tokenizer for x:
        self.tokenizer = Tokenizer()
        # fit the tokenizer on the documents
        self.tokenizer.fit_on_texts(x)
        # integer encode documents
        x = self.tokenizer.texts_to_matrix(x, mode='count')
        
        self.model = tree.DecisionTreeClassifier(max_depth = 5)
        self.model.fit(x, y)
        
    def predict(self, x):
        x = self.tokenizer.texts_to_matrix(x , mode='count')
        result = self.model.apply(x) 
        return result