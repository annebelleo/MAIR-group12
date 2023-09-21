from keras.preprocessing.text import Tokenizer
import data_preparation as dp
import pickle
import os
def get_trained_tokenizer(path_tokenizer="res/tokenizer.pickle", data_path=None):
    if os.path.exists(path_tokenizer):
        with open(path_tokenizer, 'rb') as handle:
            tokenizer = pickle.load(handle)   
            return tokenizer
        
    tokenizer = Tokenizer()
    if data_path:
        data = dp.get_data(data_path)
    else:
        data = dp.get_data()

    tokenizer.fit_on_texts(data["sentence"])
    with open(path_tokenizer, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer

