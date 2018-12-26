f1 = open('D:\TestSamlib\Test\Buddha.txt', 'r')
f2 = open('D:\TestSamlib\Test\DICTBuddha.txt', 'w')
for line in f1:
    #print(line)
    word = line.split(',')[0]
    #print(word)
    if len(word) > 2 and word.isupper():
        f2.write(word + '\n')
        print(word)
    else:
        #print(word)
        word = line.split(' ')[0]
        if len(word) > 2 and word.isupper():
            f2.write(word + '\n')
            print(word)
f1.close()
f2.close()
