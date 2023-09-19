from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.is_fitted = False
is_fitted = False

def fit (x):
    tokenizer.fit_on_texts(x)
    tokenizer.is_fitted = True

def get_tokenized(x, fit_on_texts = False):
    if not tokenizer.is_fitted or fit_on_texts:
        fit(x)
    return tokenizer.texts_to_matrix(x, mode='count')

def get_size():
    return len(tokenizer.word_index) + 1