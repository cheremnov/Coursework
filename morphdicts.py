import pymorphy2
import string
morph = pymorphy2.MorphAnalyzer()
f1 = open('D:\TestSamlib\DICTТехническое.txt', 'r')
f2 = open('D:\TestSamlib\MDICTТехническое.txt', 'w')
for line in f1:
    for word in line.split():
        # print(re.sub(r"[!?().]+$", '', word))
        # re.sub('"', '', word)
        word = word.rstrip(string.punctuation)
        word = word.lstrip(string.punctuation)
        # word = word.strip('\"')
        # print(word)
        # print(word.rstrip(string.punctuation))
        # print(re.sub(r'\p{P}+$','', word))
        f2.write(morph.parse(word)[0].normal_form + ' ')
f1.close()
f2.close()