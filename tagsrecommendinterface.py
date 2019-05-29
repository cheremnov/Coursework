import sklearn.metrics.pairwise
import os
import pandas as pd
import operator
import numpy as np
from PyQt5       import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QBrush, QColor
from functools import reduce
import sys
import re

def distances(d):
    df = pd.DataFrame(data=d)
    # cosarr = 1 - sklearn.metrics.pairwise.cosine_similarity(df.T)
    # cosarr = sklearn.metrics.pairwise.manhattan_distances(df.T)
    # cosarr = sklearn.metrics.pairwise.euclidean_distances(df.T)
    dken = len(list(d.keys()))
    cosarr = np.zeros((dken, dken))
    for i in range(dken):
        for j in range(dken):
            keyi = list(d.keys())[i]
            keyj = list(d.keys())[j]
            cosarr[i][j] = np.dot(np.array(d[keyi]), np.array(d[keyj]))
        cosarr[i][i] = 0
    return cosarr
def distance_index(d, fake_cosarr, text_index):
    dken = len(list(d.keys()))
    fake_cosarr[text_index] = np.zeros((dken))
    keyi = list(d.keys())[text_index]
    for j in range(dken):
        keyj = list(d.keys())[j]
        #print("DIST", d[keyi], d[keyj], np.dot(np.array(d[keyi]), np.array(d[keyj])))
        fake_cosarr[text_index][j] = np.dot(np.array(d[keyi]), np.array(d[keyj]))
    #print(fake_cosarr[text_index])
    fake_cosarr[text_index][text_index] = 0
    #print(fake_cosarr[text_index])
    return fake_cosarr
def recommend(cosarr, a):
    # For the first time we choose texts that we like
    # f = open('D:\TestFlibusta\Recommend\\TagsRecommend.txt', 'w')
    goodtexts = []
    goodtexts.append(a)
    # print(list(d.keys())[a])
    if (list(d.keys())[a] == "VECT172763.txt"):
        print(a)
    # For the second time we input text that we don't like
    badtexts = []
    # Just simple ratings
    ratings = {}
    print("GOOD", goodtexts)
    for curtext in range(0, len(list(d.keys()))):
        if curtext not in goodtexts and curtext not in badtexts:
            ratings[curtext] = 0
            #if cosarr[a][curtext] != 0:
            #    print("IT PHONES")
            for a in goodtexts:
                ratings[curtext] += cosarr[a][curtext] * cosarr[a][curtext]
            for a in badtexts:
                ratings[curtext] -= cosarr[a][curtext] * cosarr[a][curtext]
        curtext += 1
    #ratings = cosarr[a]
    print(ratings)
    try:
        recommended_text = max(ratings.items(), key=operator.itemgetter(1))[0]
        print(recommended_text)
        return recommended_text
    except ValueError:
        print("Ok, something wrong")
        return 0

class Widget(QtWidgets.QWidget):
    def __init__(self, books_list):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)

        self.list_of_clicked = []
        self.listView = QtWidgets.QListView()
        font = QtGui.QFont()
        #font.setFamily(_fromUtf8("FreeMono"))
        font.setBold(True)
        font.setPixelSize(20)
        self.label = QtWidgets.QLabel("Please Select item in the QListView")
        self.label.setFont(font)
        lay.addWidget(self.listView)
        lay.addWidget(self.label)

        self.entry = QtGui.QStandardItemModel()
        self.listView.setModel(self.entry)
        self.fake_cosarr = {}
        self.books_list = books_list
        self.listView.clicked[QtCore.QModelIndex].connect(self.on_clicked)
        # When you receive the signal, you call QtGui.QStandardItemModel.itemFromIndex()
        # on the given model index to get a pointer to the item

        for text in books_list:
            it = QtGui.QStandardItem(text)
            self.entry.appendRow(it)
        self.itemOld = QtGui.QStandardItem("text")
    def set_cosarr(self, cosarr):
        self.cosarr = cosarr
    def set_d(self, d):
        self.d = d
    def on_clicked(self, index):
        item = self.entry.itemFromIndex(index)
        self.label.setText("on_clicked: itemIndex=`{}`, itemText=`{}`"
                           "".format(item.index().row(), item.text()))
        self.list_of_clicked.append(item.text())
        self.label.setText(reduce(lambda x, y: str(x) + ' ' + str(y), self.list_of_clicked))
        #print(item.index().row())
        self.fake_cosarr = distance_index(self.d, self.fake_cosarr, item.index().row())
        recommended_book = recommend(self.fake_cosarr, item.index().row())
        print("RECOMMEND", recommended_book)
        self.label.setText("Рекомендуемый текст:" + self.books_list[recommended_book])
        item.setForeground(QBrush(QColor(255, 0, 0)))
        self.itemOld.setForeground(QBrush(QColor(0, 0, 0)))
        self.itemOld = item
d = {}
#directory = "D:\TestSpacebattlesPost\Vectors"
directory = "D:\\TestFlibusta\\VECT"
flib_direct = "D:\\TestFlibusta\\"
books_list = []
for file in os.listdir(directory):
    if file.endswith(".txt") and file[:4] == 'VECT':
        filename = os.path.join(directory, file)
        f = open(filename, "r")
        s = f.readline()
        d[file] = [int(c) for c in s.split() if c.isdigit()]
        #print(d[file])
        f.close()
        text_number = file[4:].split('.')[0]
        info_f = open(flib_direct+text_number+"INFO.txt", "r")
        info_text = info_f.read()
        info_f.close()
        true_info = re.sub('\n', ' ', info_text)
        books_list.append(true_info)

#cosarr = distances(d)
#print(cosarr)
#fake_cosarr = {}
#distance_index(d, fake_cosarr, 0)
#print(recommend(fake_cosarr, 0))
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Widget(books_list)
    #w.set_cosarr(cosarr)
    w.set_d(d)
    w.show()
    sys.exit(app.exec_())
0/0
