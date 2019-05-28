from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics
import os
import pandas as pd
from scipy.cluster.hierarchy import ward, dendrogram, linkage
import matplotlib.pyplot as plt



#from sklearn.metrics.pairwise import euclidean_distances
#corpus = ['For the sake for the might of our lord. For the name of the holy.', 'For the sake of our sword. Gave our lives so boldly. геймер дока 2 нуб нуб нуб']
categories = []
for file in os.listdir("D:\TestSamlib"):
    if file.endswith(".txt"):
        if file[:4] == "DICT":
            filename = os.path.join("D:\TestSamlib", file)
            f = open(filename, 'r')
            categories.append(f.read().split(' '))
            f.close()
categories_len = len(categories)
cycle_num = 0
cycle_mem = 200
d = {}
while cycle_num < 7:
    temp_constr = 0
    #d = { 'text1': [0] * categories_len, 'text2': [0] * categories_len}
    corpus = []
    for file in os.listdir("D:\TestSamlib"):
        if file.endswith(".txt"):
            if file[:7] == "MORPHED":
                if temp_constr < (cycle_num + 1) * cycle_mem:
                    if temp_constr > cycle_num * cycle_mem:
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
                    if j >= cycle_num * cycle_mem and j < (cycle_num + 1) * cycle_mem:
                        d[dict_key][i] += features[j - cycle_num * cycle_mem][vectorizer.vocabulary_[word]]
                    #print(dict_key)
                    #print(i)
                    #print(d[dict_key][i])
                    #print(word)
                    #print("END")
                    j += 1
            i += 1
    cycle_num += 1
#print(d)
df = pd.DataFrame(data=d)
#print(df)
dist = 1 - sklearn.metrics.pairwise.cosine_similarity(df.T)
linkage_matrix = linkage(dist, method = 'ward')
currtext = []
print("IWH")
a = int(input())
i = 0
lenkey = len(list(d.keys()))
print(lenkey)
currtext = []
erasetext = []
a = input()
while a != "End":
    currtext.append(int(a))
    erasetext.append(int(a))
    print(list(d.keys())[int(a)])
    a = input()
print("OKQ")
i = 0
res = 0

while i < len(linkage_matrix) and len(erasetext) > 0:
    if linkage_matrix[i][0] in erasetext:
        res = i
        erasetext.pop(erasetext.index(linkage_matrix[i][0]))
    if linkage_matrix[i][1] in erasetext:
        res = i
        erasetext.pop(erasetext.index(linkage_matrix[i][1]))
    i += 1
print("erased: ok")
print(res)
#if res > len(linkage_matrix):

 #   res = linkage_matrix
while linkage_matrix[i][0] != a and linkage_matrix[i][1] != a:
    i += 1
    print(linkage_matrix[i][0])
    print(linkage_matrix[i][1])
    print("\n")
res = 0
if linkage_matrix[i][0] == a:
    res = linkage_matrix[i][1]
else:
    res = linkage_matrix[i][0]
print("Start hierarhical rubricatinh")
print(res)
while res >= lenkey:
    print(linkage_matrix[int(res-lenkey)][0])
    print(linkage_matrix[int(res-lenkey)][1])
    print(res)
    res = linkage_matrix[int(res-lenkey)][0]
    print('\n')
print("OK")
print(res)
print(list(d.keys())[int(res)])