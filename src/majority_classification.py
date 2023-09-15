import numpy as np
import pandas as pd


class baseline_majority:
    def model_train(self, train_labels):
        '''
        train_set: array_like structure like numpy array or pandas dataframe
        class_col: index of column containing training classes
        '''
        train_labels = np.asarray(train_labels)[:]
        classif_unique, classif_occurence = np.unique(train_labels, return_counts=True)
        self.majority_class = classif_unique[np.argmax(classif_occurence)]
        self.trained = True
    
    def model_predict(self, test_set):
        if not self.trained:
            print("please train model first")
            
        else:
            test_set = np.asarray(test_set)[:]
            predictions = np.full_like(test_set, self.majority_class)
            return predictions
       
    
    def __init__(self):
        self.majority_class = -1
        self.trained = False