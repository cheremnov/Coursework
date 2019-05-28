from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics
import os
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import ward, linkage, dendrogram
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
name_texts = []
corpus = []
MAX_TEXTS = 600
for file in os.listdir("D:\TestSamlib"):
    if file.endswith(".txt"):
        if file[:7] == "MORPHED":
            if temp_constr < MAX_TEXTS:
                if temp_constr > 1:
                    name_texts.append(file)
                    filename = os.path.join("D:\TestSamlib", file)
                    f = open(filename, 'r')
                    corpus.append(f.read())
                    f.close()
                temp_constr += 1

#f = open('D:\TestSamlib\TestOutput.txt', 'r')
#for line in f:
 #   corpus.append(line)
#f.close()
vectorizer = CountVectorizer()
tmp = vectorizer.fit_transform(corpus)

tmp_metred = sklearn.metrics.pairwise.euclidean_distances(X = tmp)

#print (tmp_metred)
#print (100 * '*')

from sklearn.manifold import MDS
mds = MDS(n_components = 2, dissimilarity = 'precomputed', random_state = 1)
pos = mds.fit_transform(tmp_metred)
xs, ys = pos[ : , 0], pos[ : , 1]
#print('pos == ', pos)
#print(100*'-')
#print('xs == ', xs)
#print(100*'-')
#print('ys == ', ys)
plt.plot(xs, ys, 'b+')
plt.show()
plt.close()
#print(categories)