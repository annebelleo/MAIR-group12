

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_preparation
# lets do some EDA
#bag_o_words = dialogue_df['tokenized'].explode('tokenized')
#bag_value_counts = bag_o_words.value_counts()
#bag_o_words_labels = dialogue_df[['class', 'tokenized']].explode('tokenized')
#bag_o_words_labelcounts = bag_o_words_labels.value_counts()

#plt.plot()
#plt.bar(x = bag_value_counts.index[:20], height = bag_value_counts[:20], log = True)
#plt.xticks(rotation=45, ha='right')
#plt.show()

imgPath = 'figs/' # Set this correct

sentence_tokenize = lambda series: series.strip().split(' ')

def get_bagOWords(series : pd.Series):
    bag = series.explode()
    return bag


def plotTokenFrequency(sentences: pd.Series, top = 20):
    tokenvalcounts = sentences.apply(sentence_tokenize).explode().value_counts()
    #print(bag_of_words)
    x = tokenvalcounts.index[:top]
    y = tokenvalcounts[:top]
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_title('title')
    ax.set_xlabel('x-label')
    ax.tick_params(axis = 'x', rotation=45)
    plt.savefig(imgPath + 'token_freqs')
    

def plotTokenFrequencyPerClass(frame : pd.DataFrame):
    frame['tokenized'] = frame['sentence'].apply(sentence_tokenize)
    frame = frame.explode('tokenized')[['label', 'tokenized']]
    frame = frame.groupby(by=["label", "tokenized"], as_index=False).value_counts()
    #print(frame.loc[frame['label'] == 0])

def plotSentenceFrequency(sentences: pd.Series, top = 8):
    '''
    Needs different presentation
    '''
    sentvalcounts = sentences.apply(sentence_tokenize).value_counts()
    #print(bag_of_words)
    plt.plot()
    plt.bar(x = sentvalcounts.index[:top], height = sentvalcounts[:top], log = True)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(imgPath + 'sentence_freqs')

def plotLabelFrequency(labels : pd.Series):
    labelfrequencies = labels.value_counts()
    x = labelfrequencies.index[:]
    y = labelfrequencies[:]
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_title('Label Occurrence in Dataset')
    ax.set_xlabel('x-label')
    ax.set_xticks(x)
    ax.tick_params(axis = 'x', rotation=45)
    plt.savefig(imgPath+'label_freqs')
    
def plotModelPerformance(modelresultscollection, measure = 'accuracy', title = 'title', xlabel = '', ylabel = ''):
    x = list(modelresultscollection.keys())
    y = []
    for model in modelresultscollection.keys():
        y.append(modelresultscollection[model][measure])
    
    print(x)
    print(y)
    
    fix, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    plt.savefig(imgPath+title)
    return 0

def plotModelPerformance2(data : pd.DataFrame, 
                          index = 'Iteration',
                          model_col = 'Model',
                          measure_col = 'accuracy', 
                          title = 'title', 
                          xlabel = '', ylabel = '',
                          img_name : str = 'model_performance',
                          img_path = 'figs/'):

    df_wide = data.pivot(index=index, columns=model_col, values=measure_col)
    print(df_wide)
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(df_wide, labels=df_wide.columns)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=6.6, ha='right')
    plt.savefig(img_path+img_name)
    
    return 0

def tableLabelFreqs(precision = 3, drop_duplicates = False, img_name = 'table_labelfrequencies', table_path = 'figs/'):
    dialogDF = data_preparation.get_data(drop_duplicates=drop_duplicates)
    label_frequency = dialogDF['label'].value_counts(sort = True, normalize = True).round(decimals=precision)
    label_frequency_cumu = dialogDF['label'].value_counts(sort = True, normalize = True).cumsum().round(decimals=precision)
    labelframe = pd.concat([label_frequency, label_frequency_cumu], axis = 1, ignore_index=False)
    labelframe.columns = ['frequency', 'cumulative']
    labelframe['label'] = labelframe.index
    labelframe = labelframe[['frequency', 'cumulative']].transpose()
    
    labelframe.to_csv(path_or_buf=table_path+img_name,index_label='label')
    
    return 0



if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    
    dialogDF = data_preparation.get_data(drop_duplicates=False)
    dialogTrain, dialogTest = train_test_split(dialogDF, test_size=0.15, random_state=42)
    #print(dialogDF)
    tableLabelFreqs(dialogDF)
    raise NotImplementedError
    
    #print(bag_valueCounts)
    plotTokenFrequency(dialogDF['sentence'])
    #plotSentenceFrequency(dialogDF['sentence'])
    plotLabelFrequency(dialogDF['label'])
    #plotTokenFrequencyPerClass(dialogDF) #wip
    df = pd.DataFrame([
        [1, 'mod1', 0.1321],
        [1, 'mod2', 0.4242],
        [2, 'mod1', 0.954313],
        [3,'mod1', 0.923135],
        [2, 'mod2', 0.9231235],
        [3, 'mod2', 0.435],
        ],
        columns=['Col1', 'Col2', 'Col3'])
    print(df)
    plotModelPerformance2(df, index = 'Col1',measure_col='Col3', model_col='Col2',
                          title='Henlo', img_name='test_model_performance')
    #plotModelPerformance2(model_col='')


