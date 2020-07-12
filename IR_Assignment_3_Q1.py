# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 23:21:48 2019

@author: M.Hamza Ashraf
"""

import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models import Word2Vec
import nltk
import time

start_time = time.time()



docf = open(r"E:\FAST\7th_Semester\Information Retrieval\Assignments\Assignment_3\Datasets\Question1.txt","r")
tokenf = open(r"E:\FAST\7th_Semester\Information Retrieval\Assignments\Assignment_3\Datasets\tokens.txt","w")

text = docf.read()

new_tokens = []
tokens = []
string = ""

text = remove_stopwords(text)

sentences = []

sentences = nltk.sent_tokenize(text)

for item in sentences:
    tokens.append(list(gensim.utils.tokenize(item, lowercase=False, deacc=True)))


model_Word2Vec = Word2Vec(tokens, size=100, window=5, min_count=1, workers=4)

similar_Clean = model_Word2Vec.most_similar(positive=['Clean'])
similar_Unclean = model_Word2Vec.most_similar(positive=['Unclean'])
similar_Amazed = model_Word2Vec.most_similar(positive=['Amazed'])
similar_friendly = model_Word2Vec.most_similar(positive=['friendly'])

print("\nSimilar words for Clean : \n")

for items in similar_Clean:
    print(items)

print("\nSimilar words for Unclean : \n")

for items in similar_Unclean:
    print(items)


print("\nSimilar words for Amazed : \n")

for items in similar_Amazed:
    print(items)


print("\nSimilar words for friendly : \n")

for items in similar_friendly:
    print(items)


docf.close()
tokenf.close()

print("--- %s seconds ---" % (time.time() - start_time))