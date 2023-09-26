from keras.preprocessing.text import Tokenizer
import pickle
import os
import numpy as np
def get_tokenizer(path_tokenizer="res/tokenizer.pickle"):
    if os.path.exists(path_tokenizer):
        with open(path_tokenizer, 'rb') as handle:
            return pickle.load(handle)
        
    return Tokenizer(oov_token="unk")


def train(tokenizer, data, path_tokenizer="res/tokenizer.pickle"):
    tokenizer.fit_on_texts(data["sentence"], )
    with open(path_tokenizer, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

if __name__ == '__main__':
  tokenizer = get_tokenizer()
  test = tokenizer.texts_to_matrix( ["äöä öäö"], mode='count')
  print(np.max(test[0]))
  print(test)
