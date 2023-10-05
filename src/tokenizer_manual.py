from collections import defaultdict
import pandas as pd
import numpy as np

class Tokenizer_Manual:
  unknowntoken = '/unk/'
  
  def train(self, training_sentences : pd.Series):
    training_sentences = training_sentences.tolist()
  
    token_frequencies = defaultdict(int)
    for sent in training_sentences:
      for token in sent.split():
        token_frequencies[token] += 1
    token_frequencies[self.unknowntoken] = 0 # token for unknown words

    
    token_frequences = dict(sorted(token_frequencies.items(), key=lambda x: x[1], reverse = True))
    
    self.token2int = {k: v for v, k in enumerate(token_frequences.keys())}
    self.int2token = {v: k for k, v in self.token2int.items()}
    self.input_shape = len(self.token2int)
    self.initialized = True
    
  def map_bag(self, input_sentences : list):
    if not self.initialized:
      raise Exception("Please initialize tokenizer before usage")
    
    def vectorize_sent(sentence : str, dictionary_length : int):
      wordcount_array = np.zeros(shape = dictionary_length, dtype = np.int16)
      for token in sentence.split():
        if token not in self.token2int:
          token = self.unknowntoken
        wordcount_array[self.token2int[token]] += 1
        
      return wordcount_array
    
    if type(input_sentences) == str:
      return vectorize_sent(sentence = input_sentences, dictionary_length=self.input_shape)
    else:
      arr = np.zeros(shape = (input_sentences.shape[0], self.input_shape), dtype = np.int16)
      for i in range(len(input_sentences)):
        arr[i,:] = vectorize_sent(input_sentences[i], self.input_shape)
      #v_vectorize_sent = np.vectorize(vectorize_sent)
      return arr#v_vectorize_sent(input_sentences)
      
  def get_train_mapping(self):
    # Return an array with words at the indices where their counts would be stored.
    raise NotImplementedError
  
  def get_wordcount_dict(self):
    # returns a dictionary with wordcounts
    wc_dict = {}
    raise NotImplementedError
    return wc_dict
    
    
  
  def __call__(self, input_sentences : str):
    return self.map_bag(input_sentences)
  
  def __init__(self):
    self.tokens2int = {}
    self.int2token = {}
    self.input_shape = 0
    self.initialized = False
    
if __name__ == '__main__':
  import data_preparation
  from sklearn.model_selection import train_test_split
  
  dialogDF = data_preparation.get_data(drop_duplicates=False)
  dialogTrain, dialogTest = train_test_split(dialogDF, test_size=0.15, random_state=42)
  
  tok = Tokenizer_Manual()
  tok.train(dialogTrain['sentence'])
  dialogTrain['tokenized'] = dialogTrain.apply(lambda df: tok(df['sentence']), axis = 1)
  dialogTest['tokenized'] = dialogTest.apply(lambda df: tok(df['sentence']), axis = 1)
  