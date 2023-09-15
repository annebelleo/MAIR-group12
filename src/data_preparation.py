# By MatthijsvL
import pandas as pd

### deprecated: included in main
descriptions = ['ack','affirm','bye','confirm','deny','hello',
                  'inform','negate','null', 'repeat', 'reqalts',
                    'reqmore', 'request', 'restart','thankyou']
descript2class = {k: v for v, k in enumerate(descriptions)}
class2descript = {v: k for k, v in descript2class.items()}

def get_data(path_dialog_acts = 'res/dialog_acts.dat', drop_duplicates = False): 
  
  dialogue_df = pd.DataFrame(columns = ['label', 'sentence'])
  dialogue_df = pd.read_csv(path_dialog_acts, header = None)
  dialogue_df.insert(0, 'label', None)
  dialogue_df.columns = ['label', 'sentence']

  # apply functions
  dialog_class = lambda frame : descript2class[frame['sentence'].split(' ', maxsplit = 1)[0]]
  dialog_clean = lambda frame: frame['sentence'].split(' ',maxsplit = 1)[1].strip()
  #dialog_tokenize = lambda frame: frame['sentence'].strip().split(' ')

  dialogue_df['label'] = dialogue_df.apply(dialog_class, axis = 1)
  dialogue_df['sentence'] = dialogue_df.apply(dialog_clean, axis = 1)
  #dialogue_df['tokens'] = dialogue_df.apply(dialog_tokenize, axis = 1)
  if drop_duplicates:
    dialogue_df.drop_duplicates(subset='sentence', keep = 'first', inplace=True)

  return dialogue_df



