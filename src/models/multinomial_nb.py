# Based on this Tutorial: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#loading-the-20-newsgroups-dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class Multinomial_NB:
    def __init__(self,train):
        self.pipeline_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
        self.pipeline_clf.fit(train["lines"],train["class"])
    def prediict(self,test):
        return self.pipeline_clf.predict(test["lines"])


