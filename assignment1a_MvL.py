# By MatthijsvL
import csv
import pandas as pd
import nltk
nltk.download('punkt')
dialogueDF = pd.DataFrame(columns = ['class', 'line'])#, dtype = {'class': str, 'line' : str})
descriptions = ['ack','affirm','bye','confirm','deny','hello',
                'inform','negate','null', 'repeat', 'reqalts',
                  'reqmore', 'request', 'restart','thankyou']
descript2class = {k: v for v, k in enumerate(descriptions)}
class2descript = {v: k for k, v in descript2class.items()}
print(descript2class)
print(class2descript)

dialogueDF = pd.read_csv('dialog_acts.dat', header = None)
dialogueDF.insert(0, 'class', None)
dialogueDF.columns = ['class', 'lines']
dialog_class = lambda frame : descript2class[frame['lines'].split(' ', maxsplit = 1)[0]]
dialog_clean = lambda frame: frame['lines'].split(' ',maxsplit = 1)[1].strip()

dialogueDF['class'] = dialogueDF.apply(dialog_class, axis = 1)
dialogueDF['lines'] = dialogueDF.apply(dialog_clean, axis = 1)


dialogueDF['tokenized'] = dialogueDF.apply(lambda row: nltk.word_tokenize(row['lines'
]), axis = 1)
print(dialogueDF)

