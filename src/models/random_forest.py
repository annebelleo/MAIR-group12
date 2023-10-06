import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForest():

    def __init__(self, max_depth = None):
            # Model: Random Forest Classifier
        self.model = model_random_forest_ensemble = RandomForestClassifier(
            bootstrap=True,
            class_weight='balanced',
            criterion='gini', # No change in performance, only compute time
            n_jobs=-1, # Parallelization, faster compute time
            oob_score=True
        )
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    

    def predict(self, X_test):
        result = self.model.predict(X_test) 
        return result
