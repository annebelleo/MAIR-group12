import numpy as np
import pandas as pd
import sklearn
import m_feed_forward
import m_decision_tree
import data_preparation
from sklearn.model_selection import train_test_split

### Deprecated: included in main

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

    classif_report = sklearn.metrics.classification_report(y_true = test_labels,
                                                           y_pred=predicted_labels, 
                                                           labels=labels,
                                                           output_dict=True)
    return classif_report


if __name__ == "__main__":
    # Example line
    #model_evaluate_accuracy(dialoguePredictions_baseline[:,], dialogueTest.iloc[:,0])


    # load data:
    df = data_preparation.get_data(path_dialog_acts = "res/dialog_acts.dat", drop_duplicates=False)

    # split it to test and train:
    x, y = df["lines"], df["class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # use Decision Tree
    dt = m_decision_tree.DecisionTree(x_train, y_train)
    dt_result = dt.predict(x_test)
    print(f'descision tree result labels: {dt_result}')

    # use Feed Forward Network
    ffn = m_feed_forward.FeedForwardNetwork(x_train, y_train, epochs=10)
    ffn_result = ffn.predict(x_test)
    print(f'ffn result labels: {ffn_result}')

    # get data without duplicates:
    df_unique = data_preparation.get_data(path_dialog_acts = "res/dialog_acts.dat", drop_duplicates=True)
