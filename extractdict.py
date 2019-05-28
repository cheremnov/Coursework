f1 = open('D:\TestSamlib\StrangeFlibustaDicts\Sailor.txt', 'r')
f2 = open('D:\TestSamlib\StrangeFlibustaDicts\DICTSailor.txt', 'w')
for line in f1:
    new_line = line.split("–")[0]
    print(new_line)
    f2.write(new_line + '\n')
f1.close()
f2.close()