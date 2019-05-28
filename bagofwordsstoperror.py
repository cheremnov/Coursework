from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics
import os
import pandas as pd
from scipy.cluster.hierarchy import ward, dendrogram
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
linkage_matrix = ward(dist)
fig, ax = plt.subplots(figsize=(15, 20)) # set size
titles = list(d.keys())
ax = dendrogram(linkage_matrix, orientation="right", labels=titles)

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
   which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout
#plt.show()
plt.savefig('D:\TestSamlib\Cluster\ward_clusters.png', dpi=200)
plt.close()
#print(categories)