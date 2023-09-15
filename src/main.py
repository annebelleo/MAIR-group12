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

import visualization

descriptions = ['ack','affirm','bye','confirm','deny','hello',
                  'inform','negate','null', 'repeat', 'reqalts',
                    'reqmore', 'request', 'restart','thankyou']
descript2class = {k: v for v, k in enumerate(descriptions)}
class2descript = {v: k for k, v in descript2class.items()}

def get_data(path_dialog_acts = 'res/dialog_acts.dat', drop_duplicates = False): 
  
  dialogue_df = pd.DataFrame(columns = ['label', 'sentence'])
  dialogue_df = pd.read_csv(path_dialog_acts, header = None)
  dialogue_df.insert(0, 'label', None)
  dialogue_df.columns = ['label', 'sentence']

  # apply functions
  dialog_class = lambda frame : descript2class[frame['sentence'].split(' ', maxsplit = 1)[0]]
  dialog_clean = lambda frame: frame['sentence'].split(' ',maxsplit = 1)[1].strip()
  #dialog_tokenize = lambda frame: frame['sentence'].strip().split(' ')

  dialogue_df['label'] = dialogue_df.apply(dialog_class, axis = 1)
  dialogue_df['sentence'] = dialogue_df.apply(dialog_clean, axis = 1)
  #dialogue_df['tokens'] = dialogue_df.apply(dialog_tokenize, axis = 1)
  if drop_duplicates:
    dialogue_df.drop_duplicates(subset='sentence', keep = 'first', inplace=True)

  return dialogue_df

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

def perform_EDA(dataframe : pd.DataFrame):
   return 0
   

if __name__ == "__main__":

    seed(42) # Fix random seed

    
    ### Prepare datasets

    dialogDF = get_data(drop_duplicates=False)
    dialogTrain, dialogTest = train_test_split(dialogDF, test_size=0.15, random_state=42)

    dialogDF_nodup = get_data(drop_duplicates=True)
    dialogTrain_nodup, dialogTest_nodup = train_test_split(dialogDF, test_size=0.15, random_state=42)

    ### Perform Classification

    model_results = {}
    model_results_nodup = {}

    # Baseline: majority classification
    modelMajorityClassif = majority_classification.baseline_majority()
    modelMajorityClassif.model_train(dialogTrain['label'])
    predsMajorityClassif = modelMajorityClassif.model_predict(dialogTest['label'])
    resultsMajorityClassif = model_evaluate(predicted_labels = predsMajorityClassif, 
                                            test_labels = dialogTest['label'])
    
    model_results['majority_classif'] = resultsMajorityClassif
    
    # Baseline: majority classification no duplicates
    modelMajorityClassif_nodup = majority_classification.baseline_majority()
    modelMajorityClassif_nodup.model_train(dialogTrain_nodup['label'])
    predsMajorityClassif_nodup = modelMajorityClassif.model_predict(dialogTest_nodup['label'])
    resultsMajorityClassif_nodup = model_evaluate(predicted_labels = predsMajorityClassif_nodup, 
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
    num_models = 1
    print(resultsMajorityClassif.keys())
    print(model_results.keys())
    measures = ['accuracy', 'precision', 'recall', 'f1']
    print(list(model_results.keys()))
    
    # Makes barplot for accuracy of all models
    visualization.plotModelMetric(model_results, measure = measures[0], title = 'model_accuracy')






    

    


