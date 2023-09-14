import data_preparation as dp
import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree

def get_trained_decision_tree(x, y):
    # categories y:
    y = y.to_numpy().reshape(y.shape[0])
    # create the tokenizer for x:
    t = Tokenizer()
    # fit the tokenizer on the documents
    t.fit_on_texts(x)
    # integer encode documents
    x = t.texts_to_matrix(x, mode='count')
    
    clf = tree.DecisionTreeClassifier(max_depth = 5)
    clf = clf.fit(x, y)
    return clf