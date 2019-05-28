import os
import xml.etree.ElementTree
def primitive_compare(recom1, recom2):
    d1 = {}
    for line in open(recom1, 'r'):
        try:
            numbers = line.split(' ')
            d1[int(numbers[0])] = int(numbers[1])
        except ValueError:
            print("Oh, value error")
    d2 = {}
    for line in open(recom2, 'r'):
        try:
            numbers = line.split(' ')
            d2[int(numbers[0])] = int(numbers[1])
        except ValueError:
            print("Oh, value error")
    d1list = list(d1.keys())
    d2list = list(d2.keys())
    #allnum = min(len(d1list), len(d2list))
    processed_num = 0
    num = 0
    for text in d1list:
        if text in d2list:
            if d1[text] == d2[text]:
                num += 1
            else:
                print(text)
        processed_num += 1
    print(num)
    print(processed_num)
def parse_author(e):
    author_surname = e.find('.//{http://www.gribuser.ru/xml/fictionbook/2.0}last-name')
    return author_surname.text
def parse_genre(e):
    genre = e.find('.//{http://www.gribuser.ru/xml/fictionbook/2.0}genre')
    return genre.text
def who_best_compare(recom1, recom2):
    d1 = {}
    d2 = {}
    for line in open(recom1, 'r'):
        try:
            numbers = line.split(' ')
            e1 = xml.etree.ElementTree.parse("D:\\TestFlibusta\\"+str(int(numbers[0]))+".fb2")
            e2 = xml.etree.ElementTree.parse("D:\\TestFlibusta\\"+str(int(numbers[1])) + ".fb2")
            d1[(int(numbers[0]))] = (parse_author(e1) == parse_author(e2))
            d2[(int(numbers[0]))] = (parse_genre(e1) == parse_genre(e2))
        except ValueError:
            print("Oh, value error")
    tfbest = 0
    tagbest = 0
    authorall = 0
    genreall = 0
    alltexts = len(list(d1.keys()))
    for line in open(recom2, 'r'):
        try:
            numbers = line.split(' ')
            e1 = xml.etree.ElementTree.parse("D:\\TestFlibusta\\"+str(int(numbers[0]))+".fb2")
            e2 = xml.etree.ElementTree.parse("D:\\TestFlibusta\\"+str(int(numbers[1])) + ".fb2")
            equal_author = parse_author(e2) == parse_author(e1)
            equal_genre = parse_genre(e2) == parse_genre(e1)
            if(int(numbers[0]) in list(d1.keys()) and d1[int(numbers[0])] != equal_author):
                if equal_author == True:
                    tagbest += 1
                else:
                    tfbest += 1
                    #print(numbers[0])
            else:
                if((int(numbers[0])) in list(d2.keys()) and d2[int(numbers[0])] != equal_genre):
                    if equal_genre == True:
                        tagbest += 1
                    else:
                        tfbest+= 1
                else:
                    if(equal_author == True):
                        authorall += 1
                    elif(equal_genre == True):
                        genreall += 1
                    else:
                        #print(numbers[0])
                        int(1)
        except ValueError:
            print("Oh, value error")
    print(alltexts)
    print(tfbest)
    print(tagbest)
    print(authorall)
    print(genreall)
#primitive_compare('D:\TestFlibusta\Recommend\\TfRecommend.txt', 'D:\TestFlibusta\Recommend\\TagsRecommend.txt')
#primitive_compare('D:\TestFlibusta\Recommend\\TfRecommend.txt', 'D:\TestFlibusta\Recommend\\TfScalarRecommend.txt')
#primitive_compare('D:\TestFlibusta\Recommend\\TagsEuclideanRecommend.txt', 'D:\TestFlibusta\Recommend\\TagsRecommend.txt')
who_best_compare('D:\TestFlibusta\Recommend\\TfRecommend.txt', 'D:\TestFlibusta\Recommend\\TagsRecommend.txt')
#who_best_compare('D:\TestFlibusta\Recommend\\TfTagRecommend.txt', 'D:\TestFlibusta\Recommend\\TagsRecommend.txt')
#who_best_compare('D:\TestFlibusta\Recommend\\TfRecommend.txt', 'D:\TestFlibusta\Recommend\\TfTagRecommend.txt')
#ho_best_compare('D:\TestFlibusta\Recommend\\FuzzyRecommend.txt', 'D:\TestFlibusta\Recommend\\TagsRecommend.txt')
#who_best_compare('D:\TestFlibusta\Recommend\\FuzzyRecommend.txt', 'D:\TestFlibusta\Recommend\\TfRecommend.txt')
#who_best_compare('D:\TestFlibusta\Recommend\\TagsEuclideanRecommend.txt', 'D:\TestFlibusta\Recommend\\TagsRecommend.txt')