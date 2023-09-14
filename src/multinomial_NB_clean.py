# Based on this Tutorial: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#loading-the-20-newsgroups-dataset
import data_preparation as dp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Split into train and test data
# df = dp.get_data()
# simple_df = (df[["class","lines"]])
# train, test = train_test_split(simple_df,test_size=0.15)

def multinomial_nb(train,test):
    # Actuall ML part
    pipeline_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
    pipeline_clf.fit(train["lines"],train["class"])
    pipeline_predicted = pipeline_clf.predict(test["lines"])

    return pipeline_predicted

# Plot of predicted class to actual class histograms 
# fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
# axs[0].hist(test["class"])
# axs[1].hist(pipeline_predicted)
# plt.show()

# # Evaluation
# from sklearn.metrics import accuracy_score
# print("Accuracy: " + str(accuracy_score(test["class"], pipeline_predicted)))
