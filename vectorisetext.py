import os
import functools
directory = "D:\\TestFlibusta"
tags_list = []
file_vectors = {}
vectors = {}
good = 0
good_vectors = []
for file in os.listdir(directory):
    if file.endswith(".fb2"):
        #filename = 'D:\TestFlibusta\9373.fb2'
        filename = directory + "\\" + file
        text_number = filename.split('.')[0]
        tag_file = text_number + "TAGS.txt"
        if os.path.isfile(tag_file):
            f = open(tag_file, "r")
            tags = f.read().split('\n')
            file_vectors[file.split('.')[0]] = []
            vectors[file.split('.')[0]] = []
            #file_vectors[
            if len(tags) > 0 and len(tags[0]) > 0:
                good += 1
                good_vectors.append(file.split('.')[0])
                print(tags)
            for tag in tags:
                if tag not in tags_list:
                    tags_list.append(tag)
                file_vectors[file.split('.')[0]].append(tag)
            f.close()
print(file_vectors)
print(len(tags_list))
print("DSdfs")
print(good)
for tag in tags_list:
    #for file in list(file_vectors.keys()):
    for file in good_vectors:
        if tag in file_vectors[file]:
            vectors[file].append(1)
            if(tag == "Флэшмоб 2014"):
                print("Флэшмоб")
                print(file)
        else:
            vectors[file].append(0)
num_vect = 0
for file in good_vectors:
    num_vect += 1
    print_vectors = open(directory + "\VECT\\VECT" + file + ".txt", 'w', encoding='utf-8')
    print_vectors.write(functools.reduce(lambda x, y: str(x) + ' ' + str(y), vectors[file]) + "\n")
    print_vectors.close()
print(num_vect)