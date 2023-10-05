import numpy as np
from sklearn import tree


class DecisionTree():

    def __init__(self, max_depth = None):
        self.model = tree.DecisionTreeClassifier(max_depth=max_depth,
                                                 criterion='gini',
                                                 class_weight="balanced")
        
    def train(self, X, y):
        # categories y:
        self.model.fit(X, y)
    

    def predict(self, X):
        result = self.model.predict(X) 
        return result