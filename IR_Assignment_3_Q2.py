# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 23:21:48 2019

@author: M.Hamza Ashraf
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import  MultinomialNB
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import pandas as pd
import time
start_time = time.time()


vectorizer = CountVectorizer()
tfidfvectorizer = TfidfTransformer()

docf = open(r"E:\FAST\7th_Semester\Information Retrieval\Assignments\Assignment_3\Datasets\Question2 Dataset.tsv","r", encoding='utf8')

path = r"E:\FAST\7th_Semester\Information Retrieval\Assignments\Assignment_3\Datasets\Question2 Dataset.tsv"
text=pd.read_csv(path, delimiter='\t', encoding='utf-8')

text['review']=text['review'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())


train_reviews, test_reviews, train_sentiments, test_sentiments = train_test_split(text['review'], text['sentiment'])


BOW_train = vectorizer.fit_transform(train_reviews)
BOW_test = vectorizer.transform(test_reviews)

nb = MultinomialNB()

nb.fit(BOW_train, train_sentiments)
test_score = nb.score(BOW_test, test_sentiments)

print("Naive Bayes Results, Raw Counts:")
print('BOW: test: '+ (str(100.0*test_score)+'%'))


BOW_train = tfidfvectorizer.fit_transform(BOW_train)
BOW_test = tfidfvectorizer.transform(BOW_test)

nb.fit(BOW_train, train_sentiments)
test_score = nb.score(BOW_test, test_sentiments)

print("Naive Bayes Results, TFIDF:")
print('BOW: test: '+ (str(100.0*test_score)+'%'))

print("--- %s seconds ---" % (time.time() - start_time))