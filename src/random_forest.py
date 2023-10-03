# Import Libaries
import numpy as np
import pandas as pd
import data_preparation

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score



if __name__ == '__main__':
    import data_preparation
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from tokenizer_manual import Tokenizer_Manual
    #seed(42) # Fix random seed

    dialogDF = data_preparation.get_data(drop_duplicates=False)
    dialogDF_nodup = data_preparation.get_data(drop_duplicates = True)
    print(dialogDF['label'].value_counts(sort = True, normalize = True).cumsum())
    print(dialogDF_nodup['label'].value_counts(sort = True, normalize = True))
    dialogTrain, dialogTest = train_test_split(dialogDF, test_size=0.15)#, random_state=42)
    
    tok = Tokenizer_Manual()
    tok.train(dialogTrain['sentence'])
    dialogTrain['tokenized'] = dialogTrain.apply(lambda df: tok(df['sentence']), axis = 1)
    dialogTest['tokenized'] = dialogTest.apply(lambda df: tok(df['sentence']), axis = 1)
    
    X_train = dialogTrain['tokenized'].to_numpy()
    y_train = dialogTrain['label'].to_numpy()
    
    X_test = dialogTest['tokenized'].to_numpy()
    y_test = dialogTest['label'].to_numpy()
    print(y_train[0])
    
    print('XTRAIN: ',X_train.shape)
    print('yTRAIN: ',y_train.shape)
    forest = RandomForestClassifier(
        bootstrap=True,
        class_weight='balanced',
        criterion='gini', # No change in performance, only compute time
        n_jobs=-1, # Parallelization, faster compute time
        oob_score=True
    )
    #cross_val = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    #n_scores = cross_val_score(forest, list(X_train), y_train, scoring='accuracy', cv=cross_val, n_jobs=-1, error_score='raise')
    #print(n_scores)
    #print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    forest.fit(list(X_train), y_train)
    y_pred = forest.predict(list(X_test))
    print(classification_report(y_pred, y_test, zero_division = 0.0))
    
    
        
    
    


    