from bs4 import BeautifulSoup
from sentiment import NaturalLanguageProcesser
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
#https://www.educative.io/blog/scikit-learn-cheat-sheet-classification-regression-methods#classification

positive_reviews = BeautifulSoup(open('data/positive.review').read(), features="html5lib")
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('data/negative.review').read(), features="html5lib")
negative_reviews = negative_reviews.findAll('review_text')


#model = MultinomialNB()
model = LogisticRegression()
nlp = NaturalLanguageProcesser(model)
nlp.fit(positive_reviews, negative_reviews)
nlp.score()