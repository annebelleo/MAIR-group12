# Import Libaries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from random import seed
# Import Packages
import majority_classification
#import rule_based
#import m_decision_tree # fix deprecated error
#import m_feed_forward # fix deprecated error
import data_preparation
import visualization
import model_eval


if __name__ == "__main__":

    seed(42) # Fix random seed

    
    ### Prepare datasets

    dialogDF = data_preparation.get_data(drop_duplicates=False)
    dialogTrain, dialogTest = train_test_split(dialogDF, test_size=0.15, random_state=42)

    dialogDF_nodup = data_preparation.get_data(drop_duplicates=True)
    dialogTrain_nodup, dialogTest_nodup = train_test_split(dialogDF, test_size=0.15, random_state=42)

    ### Perform Classification

    model_results = {}
    model_results_nodup = {}

    # Baseline: majority classification
    modelMajorityClassif = majority_classification.baseline_majority()
    modelMajorityClassif.model_train(dialogTrain['label'])
    predsMajorityClassif = modelMajorityClassif.model_predict(dialogTest['label'])
    resultsMajorityClassif = model_eval.model_evaluate(predicted_labels = predsMajorityClassif, 
                                            test_labels = dialogTest['label'])
    
    model_results['majority_classif'] = resultsMajorityClassif
    
    # Baseline: majority classification no duplicates
    modelMajorityClassif_nodup = majority_classification.baseline_majority()
    modelMajorityClassif_nodup.model_train(dialogTrain_nodup['label'])
    predsMajorityClassif_nodup = modelMajorityClassif.model_predict(dialogTest_nodup['label'])
    resultsMajorityClassif_nodup =  model_eval.model_evaluate(predicted_labels = predsMajorityClassif_nodup, 
                                                  test_labels = dialogTest_nodup['label'])
    model_results_nodup['majority_classif'] = resultsMajorityClassif_nodup
    
    # baseline 2

    # baseline 2 nodup

    # ML 1
    # use Decision Tree
    #dt = m_decision_tree.DecisionTree(dialogTrain['sentence'], dialogTrain['label'])
    #dt_result = dt.predict(dialogTest['sentence'])
    #print(f'descision tree result labels: {dt_result}')
    # ML 1 nodup

    # ml2 
    # use Feed Forward Network
    #ffn = m_feed_forward.FeedForwardNetwork(dialogTrain['sentence'], dialogTrain['label'], epochs=10)
    #ffn_result = ffn.predict(dialogTest['sentence'])
    #print(f'ffn result labels: {ffn_result}')
    # ml2 nodup
   
   
   # num_models = 1
   # print(resultsMajorityClassif.keys())
   # print(model_results.keys())
    measures = ['accuracy', 'precision', 'recall', 'f1']
   # print(list(model_results.keys()))
    
    # Makes barplot for accuracy of all models
    visualization.plotModelMetric(model_results, measure = measures[0], title = 'model_accuracy')






    

    


