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
#import models.feed_forward as ffn

seed(42) # Fix random seed  
    
### Prepare datasets
dialogDF = data_preparation.get_data(drop_duplicates=False)
dialogDF_nodup = data_preparation.get_data(drop_duplicates=True)

data_Xsent = dialogDF['sentence'].to_numpy()
data_y = dialogDF['label'].to_numpy()

# Config for test suite
n_splits = 5
test_proportion = 0.15
data_size = data_Xsent.shape[0]#len(dialogDF.index)

test_size = int(np.ceil(data_size * test_proportion))


test_splits = np.zeros(shape = (n_splits,data_size), dtype = bool)
for i in range(n_splits):
    test_splits[i][np.random.choice(np.arange(data_size), size = test_size, replace=False)] = True

# ResultStructures
scores_accuracy = defaultdict(list)
scores_macroF1 = defaultdict(list)
#scores_microF1 = {} # Unused: not going to go this deep in analysis



for i in range(n_splits):
    
    X_train_sent = data_Xsent[~test_splits[i]]
    y_train = data_y[~test_splits[i]]
    X_test_sent = data_Xsent[test_splits[i]]
    y_test = data_y[test_splits[i]]
    
    tokenizer = Tokenizer()
    tokenizer.train(X_train_sent)
    
    X_train_vect = tokenizer(X_train_sent)
    X_test_vect = tokenizer(X_test_sent)
    
    ### NOTE: DEBUG CODE
    #print(tokenizer('hello darkness my old friend'))
    # for i in range(20):
    #     print(X_train_sent[i])
    #     print(np.sum(X_train_vect[i]))
    #     print(X_train_vect[i])
    #print(X_train_sent[:20])
    
    
    # Baseline: Majority 
    modelMajorityClassif = majority_classification.Baseline_majority()
    modelMajorityClassif.model_train(y_train)
    y_pred = modelMajorityClassif.model_predict(y_test) # Pass along y_test solely for test data size
    resultsMajorityClassif = model_eval.model_evaluate(predicted_labels = y_pred, 
                                            test_labels = y_test)
    print('Majority label classification:')
    print(resultsMajorityClassif)
    scores_macroF1['majority_classification'].append(resultsMajorityClassif['macro avg']['f1-score'])
    scores_accuracy['majority_classification'].append(resultsMajorityClassif['accuracy'])
    
    
    # Baseline: Rule
    y_pred = rule_based.rule_baseline(list(X_test_sent))
    resultRuleBase = model_eval.model_evaluate(predicted_labels = y_pred, 
                                            test_labels = y_test)
    print('\nRule Based Classification:')
    print(resultRuleBase)
    scores_macroF1['rule_based'].append(resultRuleBase['macro avg']['f1-score'])
    scores_accuracy['rule_based'].append(resultRuleBase['accuracy'])
    
    # Model: Decision Tree
    decision_tree_model = DecisionTree()
    decision_tree_model.train(X_train_vect, y_train)
    decision_tree_model.predict(X_test_vect)
    result_decisiontree = model_eval.model_evaluate(predicted_labels=y_pred,
                                                    test_labels = y_test)
    print('\n Decision Tree Classification: ')
    print(result_decisiontree)
    scores_macroF1['decision_tree'].append(result_decisiontree['macro avg']['f1-score'])
    scores_accuracy['decision_tree'].append(result_decisiontree['accuracy'])
    
    # Model: Random Forest Classifier
    forest = RandomForestClassifier(
        bootstrap=True,
        class_weight='balanced',
        criterion='gini', # No change in performance, only compute time
        n_jobs=-1, # Parallelization, faster compute time
        oob_score=True
    )
    
    forest.fit(list(X_train_vect), y_train)
    y_pred = forest.predict(list(X_test_vect))
    #print(classification_report(y_pred, y_test, zero_division = 0.0))
    resultRandForest = model_eval.model_evaluate(predicted_labels= y_pred,
                                                 test_labels =  y_test)
    print('\nRandom Forest Classification:')
    print(resultRandForest)
    scores_macroF1['random_forest'].append(resultRandForest['macro avg']['f1-score'])
    scores_accuracy['random_forest'].append(resultRandForest['accuracy'])
    
    # Model: Naive Bayes
    multinomial_bayes = Multinomial_NB()
    multinomial_bayes.train(X_train_vect, y_train)
    y_pred = multinomial_bayes.predict(X_test_vect)
    results_naivebayes = model_eval.model_evaluate(predicted_labels=y_pred,
                                                   test_labels=y_test)
    print('\nNaiveBayes Classifier:')
    print(results_naivebayes)
    scores_macroF1['naive_bayes'].append(results_naivebayes['macro avg']['f1-score'])
    scores_accuracy['naive_bayes'].append(results_naivebayes['accuracy'])
    
    # model = ffn.get_model(input_shape = tokenizer.input_shape)
    # ffn.train(model,X_train_vect, y_train, epochs = 5)
    # ffn_result = ffn.predict(model, X_test_vect)
    
    # resultFFN = model_eval.model_evaluate(predicted_labels = ffn_result, 
    #                                               test_labels = y_test)
print()
print('accuracy: ',scores_accuracy)
print()
print('F1: ',scores_macroF1)


    
    
    
    


#dialogTrain, dialogTest = train_test_split(dialogDF, test_size=0.15, random_state=42)
#dialogTrain_nodup, dialogTest_nodup = train_test_split(dialogDF_nodup, test_size=0.15, random_state=42)

# tok = Tokenizer_Manual()
# tok.train(dialogTrain['sentence'])
# dialogTrain['tokenized'] = dialogTrain.apply(lambda df: tok(df['sentence']), axis = 1)
# dialogTest['tokenized'] = dialogTest.apply(lambda df: tok(df['sentence']), axis = 1)

