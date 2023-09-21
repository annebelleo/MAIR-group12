from keras.preprocessing.text import Tokenizer
import data_preparation as dp
tokenizer = Tokenizer()
tokenizer.is_fitted = False
is_fitted = False

def fit():
    sentences = dp.get_data()
    tokenizer.fit_on_texts(sentences["sentence"])
    tokenizer.is_fitted = True

def get_tokenized(x, fit_on_texts = False):
    if not tokenizer.is_fitted or fit_on_texts:
        fit()
    return tokenizer.texts_to_matrix(x, mode='count')

def get_size():
    if not is_fitted:
        fit()
    return len(tokenizer.word_index) + 1