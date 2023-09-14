import numpy as np
import pandas as pd
import sklearn
def model_evaluate_accuracy(predicted_labels, test_labels):
    '''
    predicted_labels: one dimensional array like
    test_labels: one dimensional array like
    Array length is required to be the same.
    '''

    predicted_labels = np.asarray(predicted_labels,dtype=np.int16)
    test_labels = np.asarray(test_labels,dtype=np.int16)
    assert predicted_labels.shape == test_labels.shape
    
    labels = np.union1d(predicted_labels, test_labels)

    classif_report = sklearn.metrics.classification_report(y_true = test_labels, y_pred=predicted_labels, output_dict=True)
    return classif_report
    

# Example line
#model_evaluate_accuracy(dialoguePredictions_baseline[:,], dialogueTest.iloc[:,0])