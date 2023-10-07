### Definition of data plotting and analysis

import pandas as pd
import matplotlib.pyplot as plt
import data_preparation


imgPath = 'figs/' # Set this correct

sentence_tokenize = lambda series: series.strip().split(' ')

def get_bagOWords(series : pd.Series):
    bag = series.explode()
    return bag


def plotTokenFrequency(sentences: pd.Series, top = 20,
                       img_name = 'plot_tokenfrequency', 
                       img_path = imgPath):
    tokenvalcounts = sentences.apply(sentence_tokenize).explode().value_counts()
    x = tokenvalcounts.index[:top]
    y = tokenvalcounts[:top]
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_title('Token Occurence in Dataset')
    ax.set_xlabel('utterance label')
    ax.set_ylabel('count')
    ax.tick_params(axis = 'x', rotation=45)
    plt.savefig(imgPath + img_name)
    

def plotLabelFrequency(labels : pd.Series,
                       img_name = 'plot_labelfrequencies', 
                       img_path = imgPath
                       ):
    labelfrequencies = labels.value_counts()
    x = labelfrequencies.index[:]
    y = labelfrequencies[:]
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_title('Label Occurrence in Dataset')
    ax.set_xlabel('x-label')
    ax.set_xticks(x)
    ax.tick_params(axis = 'x', rotation=45)
    plt.savefig(img_path+img_name)


def plotModelPerformance(data : pd.DataFrame, 
                          index = 'Iteration',
                          model_col = 'Model',
                          measure_col = 'accuracy', 
                          title = 'title', 
                          xlabel = '', ylabel = '',
                          img_name : str = 'model_performance',
                          img_path = 'figs/'):

    df_wide = data.pivot(index=index, columns=model_col, values=measure_col)
    df_wide = df_wide[['majority classification', 'rule based classification', 'feed forward network', 'decision tree','random forest ensemble', 'multinomial naive bayes']]
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylabel(measure_col)
    ax.boxplot(df_wide, #labels=df_wide.columns,
               labels = ['majority', 'rule based', 'feed forward', 'decision tree', 'random forest', 'naive bayes'])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.xticks(rotation=6.6, ha='right')
    plt.savefig(img_path+img_name)

def tableLabelFreqs(precision = 3, drop_duplicates = False, table_name = 'table_labelfrequencies', table_path = 'figs/'):
    dialogDF = data_preparation.get_data(drop_duplicates=drop_duplicates)
    label_frequency = dialogDF['label'].value_counts(sort = True, normalize = True).round(decimals=precision)
    label_frequency_cumu = dialogDF['label'].value_counts(sort = True, normalize = True).cumsum().round(decimals=precision)
    labelframe = pd.concat([label_frequency, label_frequency_cumu], axis = 1, ignore_index=False)
    labelframe.columns = ['frequency', 'cumulative']
    labelframe['label'] = labelframe.index
    labelframe = labelframe[['frequency', 'cumulative']].transpose()
    
    labelframe.to_csv(path_or_buf=table_path+table_name,index_label='label')
    
def table_result_statistics(results: pd.DataFrame, measure = 'accuracy', table_name = 'table_results', table_path = 'figs/'):
    results = results.pivot(index='Iteration', columns='Model', values=measure)[['majority classification', 'rule based classification', 'feed forward network', 'decision tree','random forest ensemble', 'multinomial naive bayes']]
    results = results.describe()
    results.columns=['majority', 'rule based', 'feed forward', 'decision tree', 'random forest', 'naive bayes']
    results = results.loc[['count', 'mean', 'std', 'max']]
    
    results.to_csv(path_or_buf=table_path+table_name,index_label='model')

