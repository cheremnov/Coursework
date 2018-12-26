from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics
from sklearn.cluster import KMeans
import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



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
df = pd.DataFrame(data=d)
npar = np.array(df)
#print(df)
dist = 1 - sklearn.metrics.pairwise.cosine_similarity(df.T)
#clust = KMeans(init = 'k-means++').fit(dist)
#reduced_data = PCA(n_components=2).fit_transform(df.T)
reduced_data = dist
kmeans = KMeans(init = 'k-means++')
clust = kmeans.fit(reduced_data)
print(clust.labels_)
tempi = 0
while tempi < len(clust.labels_):
    print(clust.labels_[tempi])
    print(list(d.keys())[tempi])
    tempi += 1
titles = list(d.keys())
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
#Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
#Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
#plt.imshow(Z, interpolation='nearest',
 #          extent=(xx.min(), xx.max(), yy.min(), yy.max()),
  #         cmap=plt.cm.Paired,
   #        aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='r', zorder=10)
plt.title('K-means clustering'
          'Centroids are marked with red cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
#tsne_init = 'random'
#tsne_perplexity = 20.0
#tsne_early_exaggeration = 4.0
#tsne_learning_rate = 1000
#random_state = 1
#model = TSNE(n_components=2, random_state=random_state, init=tsne_init, perplexity=tsne_perplexity,
 #       early_exaggeration=tsne_early_exaggeration, learning_rate=tsne_learning_rate)
#transformed_centroids = model.fit_transform(clust.cluster_centers_)
#plt.scatter(transformed_centroids[:, 0], transformed_centroids[:, 1], marker='x')
#plt.show()
#model = TSNE(n_components=2, random_state=random_state, init=tsne_init, perplexity=tsne_perplexity,
       # early_exaggeration=tsne_early_exaggeration, learning_rate=tsne_learning_rate)
#transformed = model.fit_transform(npar)
#plt.scatter(transformed[: 0], transformed[: 1])