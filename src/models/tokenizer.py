from keras.preprocessing.text import Tokenizer
import data_preparation as dp
import pickle
import os
def get_tokenizer(path_tokenizer="res/tokenizer.pickle"):
    if os.path.exists(path_tokenizer):
        with open(path_tokenizer, 'rb') as handle:
            return pickle.load(handle)
        
    return Tokenizer()


def train(tokenizer, data, path_tokenizer="res/tokenizer.pickle"):
    tokenizer.fit_on_texts(data["sentence"])
    with open(path_tokenizer, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        