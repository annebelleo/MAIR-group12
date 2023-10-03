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
    self.inputshape = len(self.token2int)
    self.initialized = True
  
  def __call__(self, sentence : str):
    if not self.initialized:
      raise Exception("Please initialize tokenizer before usage")
    
    wordcount_array = np.zeros(shape = self.inputshape, dtype = np.int64)
    for token in sentence.split():
      if token not in self.token2int:
        token = self.unknowntoken
      wordcount_array[self.token2int[token]] += 1
      
    return wordcount_array
  
  def __init__(self):
    self.tokens2int = {}
    self.int2token = {}
    self.inputshape = 0
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
  