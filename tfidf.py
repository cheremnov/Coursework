from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import os
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold # import KFold
def get_scalar(XA, XB):
    l = []
    for vecb in XB:
        scalar = np.dot(XA, vecb)
        l.append(scalar[0])
    a = np.array([l])
    return a
#stemmer = SnowballStemmer("russian")
def get_text(file):
    filename = "D:\\TestFlibusta\\" + file
    text_number = filename.split('.')[0]
    if file.endswith(".fb2") and os.path.isfile(text_number+"Tags.txt"):
        try:
            f = open(filename, 'r', encoding='utf-8')
            s = f.read()
            f.close()
            return s
        except UnicodeDecodeError:
            return ""
    return ""
def get_corpus(names):
    l = []
    for file in os.listdir("D:\TestFlibusta"):
        s = get_text(file)
        if(len(s) > 0):
            l.append(get_text(file))
            names.append(file.split('.')[0])
    return l

names = []
corpus = get_corpus(names)
stopWords = stopwords.words('russian')
vectorizer = TfidfVectorizer(analyzer = "word" , tokenizer = None , preprocessor = None , \
stop_words = stopWords , max_features = 20000)
#beerlist = np.array(['heinekin lager', 'corona lager', 'heinekin ale', 'budweiser lager'])
textlist_tfidf = vectorizer.fit_transform(corpus).toarray()
#f = open('D:\TestFlibusta\Recommend\\TfRecommend.txt', 'w')
f = open('D:\TestFlibusta\Recommend\\TfScalarRecommend.txt', 'w')
for num in range(0, len(corpus)):
    f.write(names[num] + ' ')
    text_tfidf = vectorizer.transform([corpus[num]]).toarray()
    #cosine_dist = cdist(text_tfidf, textlist_tfidf, 'cosine')
    #rec_idx = cosine_dist.argsort()
    euclidean_dist = cdist(text_tfidf, textlist_tfidf, 'euclidean')
    rec_idx = euclidean_dist.argsort()
    scalar_dist = get_scalar(text_tfidf, textlist_tfidf)
    scalar_dist = euclidean_dist * scalar_dist
    new_idx = scalar_dist.argsort()
    f.write(names[new_idx[0][1]] + '\n')
f.close()
#df = pd.DataFrame(data=d)
#dist = 1 - sklearn.metrics.pairwise.cosine_similarity(df.T)