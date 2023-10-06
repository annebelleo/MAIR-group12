from typing import Any
import numpy as np
import pandas as pd
import sklearn
#import models.feed_forward as feed_forward
#import models.decision_tree as decision_tree
import data_preparation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def model_evaluate(predicted_labels, test_labels):
    '''
    predicted_labels: one dimensional array like
    test_labels: one dimensional array like
    Array length is required to be the same.
    '''

    predicted_labels = np.asarray(predicted_labels,dtype=np.int16)
    test_labels = np.asarray(test_labels,dtype=np.int16)
    assert predicted_labels.shape == test_labels.shape
    
    labels = np.union1d(predicted_labels, test_labels)

    classif_report = classification_report(y_true = test_labels, 
                                           y_pred=predicted_labels, 
                                           output_dict=True,
                                           zero_division=0.0)
    return classif_report

class ResultsFrame:
    columns = ['Model', 
                'Iteration', 
                'Setup', 
                'accuracy',
                'macro_F1']
    def add_record(self, model_name : str, iteration_num: int, 
                   setup_description : str, classification_report : dict):
        record = pd.DataFrame([{
            self.columns[0] : model_name,
            self.columns[1] : iteration_num,
            self.columns[2] : setup_description,
            self.columns[3] : classification_report['accuracy'],
            self.columns[4] : classification_report['macro avg']['f1-score']
        }])
        self.data_frame = pd.concat([self.data_frame, record],
                                    ignore_index=True)
        
    def get_frame(self):
        return self.data_frame       
        
    
    def __init__(self):
        self.data_frame = pd.DataFrame(columns = self.columns)
        