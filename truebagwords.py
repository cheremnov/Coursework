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
MAX_TEXTS = 100
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
#print( vectorizer.fit_transform(corpus).todense() )
features = vectorizer.fit_transform(corpus).todense()
#features = vectorizer.fit_transform(corpus).toarray()
print(features)
#print(features[0][vectorizer.vocabulary_['the']])
#print( vectorizer.vocabulary_ )
df = pd.DataFrame(data=features)
#print(df)
linkage_matrix = linkage(features, method = 'complete')
print(linkage_matrix)
fig, ax = plt.subplots(figsize=(15, 20)) # set size
titles = name_texts
ax = dendrogram(linkage_matrix, orientation="right", labels=titles)

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
   which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')
plt.tight_layout() #show plot with tight layout
#plt.show()
plt.savefig('D:\TestSamlib\Cluster\ward_clusters_linkage_single.png', dpi=200)
plt.close()
#print(categories)