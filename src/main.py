# Import Libaries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from random import seed
# Import Packages
import models.majority_classification as majority_classification
#import rule_based
#import decision_tree # fix deprecated error
#import feed_forward # fix deprecated error
import data_preparation
import visualization
import model_eval
import models.decision_tree as decision_tree
import models.feed_forward as feed_forward
import models.rule_based as rule_based
if __name__ == "__main__":

    seed(42) # Fix random seed

    
    ### Prepare datasets

    dialogDF = data_preparation.get_data(drop_duplicates=False)
    dialogTrain, dialogTest = train_test_split(dialogDF, test_size=0.15, random_state=42)

    dialogDF_nodup = data_preparation.get_data(drop_duplicates=True)
    dialogTrain_nodup, dialogTest_nodup = train_test_split(dialogDF_nodup, test_size=0.15, random_state=42)
    # Rule-based
    predictionRuleBase = rule_based.rule_baseline(dialogTest["sentence"])
    resultRuleBase = model_eval.model_evaluate(predicted_labels = predictionRuleBase, 
                                            test_labels = dialogTest['label'])
    print("Rule Based results:")
    #print(f"{resultRuleBase}")
    print(f"{resultRuleBase['accuracy']}")
    print(f"{resultRuleBase['macro avg']}")
    print(f"{resultRuleBase['weighted avg']}")
    
    ### Perform Classification

    model_results = {}
    model_results_nodup = {}

    # Baseline: majority classification
    modelMajorityClassif = majority_classification.Baseline_majority()
    modelMajorityClassif.model_train(dialogTrain['label'])
    predsMajorityClassif = modelMajorityClassif.model_predict(dialogTest['label'])
    resultsMajorityClassif = model_eval.model_evaluate(predicted_labels = predsMajorityClassif, 
                                            test_labels = dialogTest['label'])
    print(f"Majority classif results:")
    #print(f"{resultsMajorityClassif}")
    print(f"{resultsMajorityClassif['accuracy']}")
    print(f"{resultsMajorityClassif['macro avg']}")
    print(f"{resultsMajorityClassif['weighted avg']}")
    model_results['majority_classif'] = resultsMajorityClassif
    
    # Baseline: majority classification no duplicates
    modelMajorityClassif_nodup = majority_classification.Baseline_majority()
    modelMajorityClassif_nodup.model_train(dialogTrain_nodup['label'])
    predsMajorityClassif_nodup = modelMajorityClassif.model_predict(dialogTest_nodup['label'])
    resultsMajorityClassif_nodup =  model_eval.model_evaluate(predicted_labels = predsMajorityClassif_nodup, 
                                                  test_labels = dialogTest_nodup['label'])
    model_results_nodup['majority_classif'] = resultsMajorityClassif_nodup
    print(f"Majority Unique classif results:")
    print(f"{resultsMajorityClassif_nodup['accuracy']}")
    print(f"{resultsMajorityClassif_nodup['macro avg']}")
    print(f"{resultsMajorityClassif_nodup['weighted avg']}")
    
    # baseline 2

    # baseline 2 nodup

    # ML 1
    # use Decision Tree
    dt = decision_tree.DecisionTree(dialogTrain['sentence'], dialogTrain['label'])
    dt_result = dt.predict(dialogTest['sentence'])
    resultDecisionTree = model_eval.model_evaluate(predicted_labels = dt_result, 
                                                  test_labels = dialogTest["label"])
    model_results['decision_tree'] = resultDecisionTree
    print(f"Desition Tree Unique classif results:")
    print(f"{resultDecisionTree['accuracy']}")
    print(f"{resultDecisionTree['macro avg']}")
    print(f"{resultDecisionTree['weighted avg']}")
    
    # ML 1 nodup
    dt_nodup = decision_tree.DecisionTree(dialogTrain_nodup['sentence'], dialogTrain_nodup['label'])
    dt_result_nodup = dt_nodup.predict(dialogTest_nodup['sentence'])
    resultDecisionTreeUnique = model_eval.model_evaluate(predicted_labels = dt_result_nodup,
                                                                     test_labels= dialogTest_nodup['label'])
    model_results_nodup['decision_tree_nodup'] = resultDecisionTreeUnique
    print(f"Desition Tree Unique classif results:")
    print(f"{resultDecisionTreeUnique['accuracy']}")
    print(f"{resultDecisionTreeUnique['macro avg']}")
    print(f"{resultDecisionTreeUnique['weighted avg']}")
    
    # ml2 
    # use Feed Forward Network
    ffn = feed_forward.FeedForwardNetwork(dialogTrain['sentence'], dialogTrain['label'], epochs=10)
    ffn_result = ffn.predict(dialogTest['sentence'])
    
    resultFFN = model_eval.model_evaluate(predicted_labels = ffn_result, 
                                                  test_labels = dialogTest["label"])
    
    model_results_nodup['ffn'] = resultFFN
    print(f"FeedForward Network:")
    print(f"{resultFFN['accuracy']}")
    print(f"{resultFFN['macro avg']}")
    print(f"{resultFFN['weighted avg']}")
    
    #print(f'ffn result labels: {ffn_result}')
    # ml2 nodup
    ffn_nodup = feed_forward.FeedForwardNetwork(dialogTrain_nodup['sentence'], dialogTrain_nodup['label'], epochs = 10)
    ffn_nodup_results = ffn_nodup.predict(dialogTest_nodup['sentence'])
    resultFFNUnique = model_eval.model_evaluate(predicted_labels = ffn_nodup_results, test_labels=dialogTest_nodup['label'])
    print(f"FeedForward Unique Network:")
    print(f"{resultFFNUnique['accuracy']}")
    print(f"{resultFFNUnique['macro avg']}")
    print(f"{resultFFNUnique['weighted avg']}")
    
   # num_models = 1
   # print(resultsMajorityClassif.keys())
   # print(model_results.keys())
    measures = ['accuracy', 'macro avg', 'weighted avg', 'f1']
   # print(list(model_results.keys()))
    
    # Makes barplot for accuracy of all models
    #visualization.plotModelMetric(model_results, measure = measures[0], title = 'model_accuracy')
    #visualization.plotModelMetric(model_results_nodup, measure = measures[0], title = 'model_accuracy_nodup')






    

    


