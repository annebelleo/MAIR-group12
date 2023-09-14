# Based on this Tutorial: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#loading-the-20-newsgroups-dataset

import data_preparation as dp
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Text tokenizing
count_vect = CountVectorizer()
df = dp.get_data()
simple_df = (df[["class","lines"]])
train, test = train_test_split(simple_df,test_size=0.15)

x_train_counts = count_vect.fit_transform(train["lines"])

# From occurences to frequencies
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

# Classifier training
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(x_train_tfidf, train["class"])

# Classifier testing
test_lines = test["lines"]
x_test_counts = count_vect.transform(test_lines)
x_test_tfidf = tfidf_transformer.transform(x_test_counts)
predicted = clf.predict(x_test_tfidf)

# All code above can be condensed using a classifier if we want it cleaner but results should be the same.
from sklearn.pipeline import Pipeline
pipeline_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
pipeline_clf.fit(train["lines"],train["class"])
pipeline_predicted = pipeline_clf.predict(test["lines"])

# All evalutaions from here on are done on the pipeline predictions. They give the same results as the verbose code :)

# Plot of predicted class to actual class histograms 
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
# axs[0].hist(predicted)
axs[0].hist(test["class"])
axs[1].hist(pipeline_predicted)
plt.show()

# Evaluation
from sklearn.metrics import accuracy_score
# print("SKLearn accuracy: " + str(accuracy_score(test["class"], predicted)))
# print("NumPY accuracy: " + str(np.mean(predicted==test["class"])))
print("Accuracy: " + str(accuracy_score(test["class"], pipeline_predicted)))
