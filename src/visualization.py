

from main import get_data, descriptions, descript2class, class2descript
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    print(frame.loc[frame['label'] == 0])

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
    
def plotModelMetric(modelresultscollection, measure = 'accuracy', title = 'title', xlabel = '', ylabel = ''):
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


    



if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    
    dialogDF = get_data(drop_duplicates=False)
    dialogTrain, dialogTest = train_test_split(dialogDF, test_size=0.15, random_state=42)
    #print(dialogDF)
    
    
    #print(bag_valueCounts)
    plotTokenFrequency(dialogDF['sentence'])
    #plotSentenceFrequency(dialogDF['sentence'])
    plotLabelFrequency(dialogDF['label'])
    #plotTokenFrequencyPerClass(dialogDF) #wip
        


