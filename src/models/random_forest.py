import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle


class RandomForest():

    def __init__(self):
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
    
    def save(self, model_path="res/models/random_forest.pkl"):
        with open(model_path, 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
                                  
    def predict(self, X_test):
        result = self.model.predict(X_test) 
        return result

def load_model(model_path="res/models/random_forest.pkl"):
    with open(model_path, 'rb') as handle:
        return pickle.load(handle)
