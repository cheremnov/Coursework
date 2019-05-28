from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics
from sklearn.cluster import DBSCAN
import os
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt



#from sklearn.metrics.pairwise import euclidean_distances
#corpus = ['For the sake for the might of our lord. For the name of the holy.', 'For the sake of our sword. Gave our lives so boldly. геймер дока 2 нуб нуб нуб']
temp_constr = 0
categories = []
#d = { 'text1': [0] * categories_len, 'text2': [0] * categories_len}
for file in os.listdir("D:\TestSamlib"):
    if file.endswith(".txt"):
        if file[:4] == "DICT":
            filename = os.path.join("D:\TestSamlib", file)
            f = open(filename, 'r')
            categories.append(f.read().split(' '))
            f.close()
categories_len = len(categories)
d = {}
corpus = []
for file in os.listdir("D:\TestSamlib"):
    if file.endswith(".txt"):
        if file[:7] == "MORPHED":
            if temp_constr < 200:
                if temp_constr > 1:
                    filename = os.path.join("D:\TestSamlib", file)
                    f = open(filename, 'r')
                    corpus.append(f.read())
                    d[file[7:]] = [0] * categories_len
                    f.close()
                temp_constr += 1

#f = open('D:\TestSamlib\TestOutput.txt', 'r')
#for line in f:
 #   corpus.append(line)
#f.close()

vectorizer = CountVectorizer()
#print( vectorizer.fit_transform(corpus).todense() )
#features = vectorizer.fit_transform(corpus).todense()
features = vectorizer.fit_transform(corpus).toarray()
print(features)
#print(features[0][vectorizer.vocabulary_['the']])
#print( vectorizer.vocabulary_ )
text_dict = list(d.keys())
#print(d)
for word in vectorizer.vocabulary_:
    i = 0
    while i < len(categories):
        if word in categories[i]:
            #WARNING: It assumes keys of the dictionary in insertion order! We append new text to corpus, we add the document name in the dictionary
            #We use j, because we are sure that dict_key is exactly text that was in the corpus
            j = 0
            for dict_key in d.keys():
                #if j == 35:
                 #   print(dict_key)
                  #  print(features[j])
                d[dict_key][i] += features[j][vectorizer.vocabulary_[word]]
                #print(dict_key)
                #print(i)
                #print(d[dict_key][i])
                #print(word)
                #print("END")
                j += 1
        i += 1
for dict_key in d.keys():
    dictname = "D:\TestSamlib\Vectors\VECT"
    dictname += dict_key
    f1 = open(dictname, "w")
    dict_i = 0
    while dict_i < categories_len:
        f1.write(str(d[dict_key][dict_i]))
        f1.write(" ")
        dict_i+=1
    f1.close()
#print(d)
df = pd.DataFrame(data=d)
npar = np.array(df)
#print(df)
dist = 1 - sklearn.metrics.pairwise.cosine_similarity(df.T)
clust = DBSCAN(eps = 0.1, min_samples = 10).fit(dist)
print(clust.labels_)
titles = list(d.keys())
#print(categories)