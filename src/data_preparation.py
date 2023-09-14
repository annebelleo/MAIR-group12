# By MatthijsvL
import csv
import pandas as pd


def get_data(path_dialog_acts = 'res/dialog_acts.dat', drop_duplicates = True): 
  
  dialogue_df = pd.DataFrame(columns = ['class', 'line'])#, dtype = {'class': str, 'line' : str})
  descriptions = ['ack','affirm','bye','confirm','deny','hello',
                  'inform','negate','null', 'repeat', 'reqalts',
                    'reqmore', 'request', 'restart','thankyou']
  descript2class = {k: v for v, k in enumerate(descriptions)}
  class2descript = {v: k for k, v in descript2class.items()}
 # print(descript2class)
 # print(class2descript)

  dialogue_df = pd.read_csv(path_dialog_acts, header = None)
  dialogue_df.insert(0, 'class', None)
  dialogue_df.columns = ['class', 'lines']
  dialog_class = lambda frame : descript2class[frame['lines'].split(' ', maxsplit = 1)[0]]
  dialog_clean = lambda frame: frame['lines'].split(' ',maxsplit = 1)[1].strip()

  dialogue_df['class'] = dialogue_df.apply(dialog_class, axis = 1)
  dialogue_df['lines'] = dialogue_df.apply(dialog_clean, axis = 1)
  if drop_duplicates:
    dialogue_df.drop_duplicates(inplace=True)

  return dialogue_df


