# Import Libaries
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from random import seed

import data_preparation
import visualization
import model_eval
from tokenizer_manual import Tokenizer_Manual as Tokenizer

# Models
import models.majority_classification as majority_classification
import models.rule_based as rule_based
from models.decision_tree import DecisionTree
from models.multinomial_nb import Multinomial_NB
from sklearn.ensemble import RandomForestClassifier
from models.random_forest import RandomForest
#import models.feed_forward as ffn

seed(42) # Fix random seed  

def EDA():
    '''exploratory data analysis'''
    dialogDF = data_preparation.get_data(drop_duplicates=False)
    visualization.plotTokenFrequency(dialogDF['sentences'])
    visualization.plotTokenFrequencyPerClass(dialogDF)
    visualization.plotLabelFrequency(dialogDF['labels'])
    


def perform_test_suite(setup_name : str,
                       test_proportion : float = 0.15,
                       n_iterations : int = 5,
                       drop_duplicate_sentences : bool = False):
    suite_results = model_eval.ResultsFrame()
    
    dialogDF = data_preparation.get_data(drop_duplicates=drop_duplicate_sentences)
    data_Xsent = dialogDF['sentence'].to_numpy()
    data_y = dialogDF['label'].to_numpy()
    
    test_proportion = test_proportion
    data_size = data_Xsent.shape[0]
    test_size = int(np.ceil(data_size * test_proportion))
    test_splits = np.zeros(shape = (n_iterations,data_size), dtype = bool)
    
    # Setup test splits for each iteration
    for iter in range(n_iterations):
        test_splits[iter][np.random.choice(np.arange(data_size), size = test_size, replace=False)] = True
    # Perform test suite
    for iter in range(n_iterations):
        iter_num = iter + 1
        
        X_train_sent = data_Xsent[~test_splits[iter]]
        y_train = data_y[~test_splits[iter]]
        X_test_sent = data_Xsent[test_splits[iter]]
        y_test = data_y[test_splits[iter]]
        
        tokenizer = Tokenizer()
        tokenizer.train(X_train_sent)
        
        X_train_vect = tokenizer(X_train_sent)
        X_test_vect = tokenizer(X_test_sent)
        
        # Baseline: Majority 
        model_majority_classif = majority_classification.Baseline_majority()
        model_majority_classif.model_train(y_train)
        y_pred = model_majority_classif.model_predict(y_test) # Pass along y_test solely for test data size
        model_classification_report = model_eval.model_evaluate(predicted_labels = y_pred, 
                                                test_labels = y_test)
        
        suite_results.add_record(model_name='majority classification',
                        iteration_num=iter_num,
                        setup_description=setup_name,
                        classification_report=model_classification_report)
        
        
        # Baseline: Rule Based Classification
        y_pred = rule_based.rule_baseline(list(X_test_sent))
        model_classification_report = model_eval.model_evaluate(predicted_labels = y_pred, 
                                                test_labels = y_test)
        
        suite_results.add_record(model_name='rule based classification',
                        iteration_num=iter_num,
                        setup_description=setup_name,
                        classification_report=model_classification_report)
        
        # Model: Decision Tree
        model_decision_tree = DecisionTree()
        model_decision_tree.train(X_train_vect, y_train)
        model_decision_tree.predict(X_test_vect)
        model_classification_report = model_eval.model_evaluate(predicted_labels=y_pred,
                                                        test_labels = y_test)
        
        suite_results.add_record(model_name='decision tree',
                        iteration_num=iter_num,
                        setup_description=setup_name,
                        classification_report=model_classification_report)
        
        # Model: Random Forest
        model_random_forest_ensemble = RandomForest()
        model_random_forest_ensemble.train(list(X_train_vect), y_train)
        y_pred = model_random_forest_ensemble.predict(list(X_test_vect))
        
        model_classification_report = model_eval.model_evaluate(predicted_labels= y_pred,
                                                    test_labels =  y_test)

        suite_results.add_record(model_name='random forest ensemble',
                        iteration_num=iter_num,
                        setup_description=setup_name,
                        classification_report=model_classification_report)
        
        # Model: Naive Bayes
        model_multinomial_bayes = Multinomial_NB()
        model_multinomial_bayes.train(X_train_vect, y_train)
        y_pred = model_multinomial_bayes.predict(X_test_vect)
        model_classification_report = model_eval.model_evaluate(predicted_labels=y_pred,
                                                    test_labels=y_test)

        suite_results.add_record(model_name='multinonial naive bayes',
                        iteration_num=iter_num,
                        setup_description=setup_name,
                        classification_report=model_classification_report)
    
    return suite_results.get_frame()

if __name__ == '__main__':
    EDA() # Performs Exploratory Data Analysis steps and makes plots.
    ### Experiment suite 1: Unpreprocessed Data
    results_base = perform_test_suite('base',
                                           test_proportion=0.15,
                                           n_iterations=5,
                                           drop_duplicate_sentences=False)
    print(results_base)
    results_nodup = perform_test_suite('nodup',
                                       test_proportion=0.15,
                                       n_iterations=5,
                                       drop_duplicate_sentences=True)
    print(results_nodup)
    # Prepare datasets


    
    
    