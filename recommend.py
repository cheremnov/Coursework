import sklearn.metrics.pairwise
import os
import pandas as pd
import numpy as np
d = {}
for file in os.listdir("D:\TestSamlib\Vectors"):
    if file.endswith(".txt") and file[:4] == 'VECT':
        filename = os.path.join("D:\TestSamlib\Vectors", file)
        f = open(filename, "r")
        s = f.readline()
        d[file] = [int(c) for c in s.split() if c.isdigit()]
        #print(d[file])
        f.close()
df = pd.DataFrame(data=d)
cosarr = 1 - sklearn.metrics.pairwise.cosine_similarity(df.T)
currtext = []
a = input()
while a != "End":
    currtext.append(int(a))
    print(list(d.keys())[int(a)])
    a = input()
mindiff = 0
alltext = 0
while alltext < 197:
    if alltext not in currtext:
        eucl1 = 0
        eucl2 = 0
        for a in currtext:
            eucl1 += cosarr[a][mindiff] * cosarr[a][mindiff]
            eucl2 += cosarr[a][alltext] * cosarr[a][alltext]
        if eucl1 > eucl2:
            mindiff = alltext
    alltext += 1
print(mindiff)
print(list(d.keys())[mindiff])