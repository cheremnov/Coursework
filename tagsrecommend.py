import sklearn.metrics.pairwise
import os
import pandas as pd
import operator
import numpy as np
d = {}
#directory = "D:\TestSpacebattlesPost\Vectors"
directory = "D:\\TestFlibusta\\VECT"
processed = 0
for file in os.listdir(directory):
    if file.endswith(".txt") and file[:4] == 'VECT':
        filename = os.path.join(directory, file)
        f = open(filename, "r")
        s = f.readline()
        d[file] = [int(c) for c in s.split() if c.isdigit()]
        #print(d[file])
        f.close()
        processed += 1
        #if processed > 100:
        #    break
print()
df = pd.DataFrame(data=d)
#cosarr = 1 - sklearn.metrics.pairwise.cosine_similarity(df.T)
#cosarr = sklearn.metrics.pairwise.manhattan_distances(df.T)
#cosarr = sklearn.metrics.pairwise.euclidean_distances(df.T)
dken = len(list(d.keys()))
cosarr = np.zeros((dken, dken))
for i in range(dken):
    for j in range(dken):
        keyi = list(d.keys())[i]
        keyj = list(d.keys())[j]
        cosarr[i][j] = np.dot(np.array(d[keyi]), np.array(d[keyj]))
    cosarr[i][i] = 0
print(cosarr)
#For the first time we choose texts that we like
#f = open('D:\TestFlibusta\Recommend\\TagsRecommend.txt', 'w')
f = open('D:\TestFlibusta\Recommend\\TagsEuclideanRecommend.txt', 'w')
for a in range(0, len(list(d.keys()))):
    goodtexts = []
    goodtexts.append(a)
    f.write(list(d.keys())[a][4:].split('.')[0] + ' ')
    #print(list(d.keys())[a])
    if(list(d.keys())[a] == "VECT172763.txt"):
        print(a)
    #For the second time we input text that we don't like
    badtexts = []
    mindiff = 0
    curtext = 0
    #Just simple ratings
    ratings = {}
    for curtext in range(0, len(list(d.keys()))):
        if curtext not in goodtexts and curtext not in badtexts:
            ratings[curtext] = 0
            for a in goodtexts:
                ratings[curtext] += cosarr[a][curtext] * cosarr[a][curtext]
            for a in badtexts:
                ratings[curtext] -= cosarr[a][curtext] * cosarr[a][curtext]
        curtext += 1
    if(a == 80):
        print(ratings)
    try:
        recommended_text = max(ratings.items(), key=operator.itemgetter(1))[0]
        f.write(list(d.keys())[recommended_text][4:].split('.')[0] + '\n')
    except ValueError:
        print("Ok, something wrong")
f.close()