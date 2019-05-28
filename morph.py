import pymorphy2
import string
import regex as re
morph = pymorphy2.MorphAnalyzer()
print(morph.parse('рябчиков')[0].normal_form)
import os
for file in os.listdir("D:\TestSamlib"):
    if file.endswith(".txt"):
        filename = os.path.join("D:\TestSamlib", file)
        if os.path.getsize(filename) > 0 and file[:7] != 'MORPHED':
            f1 = open(filename, 'r')
            filename2 = os.path.join("D:\TestSamlib", 'MORPHED' + file)
            f2 = open(filename2, 'w')
            for line in f1:
                for word in line.split():
                    #print(re.sub(r"[!?().]+$", '', word))
                    #re.sub('"', '', word)
                    word = word.rstrip(string.punctuation)
                    word = word.lstrip(string.punctuation)
                    #word = word.strip('\"')
                    #print(word)
                    #print(word.rstrip(string.punctuation))
                    #print(re.sub(r'\p{P}+$','', word))
                    f2.write(morph.parse(word)[0].normal_form + ' ')
            f1.close()
            f2.close()