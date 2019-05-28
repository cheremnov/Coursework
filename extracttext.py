from bs4 import BeautifulSoup
import os
for file in os.listdir("D:\TestSamlib"):
    if file.endswith(".html"):
        filename = os.path.join("D:\TestSamlib", file)
        if os.path.getsize(filename) > 0:
            f1 = open(filename, 'r')
            norm_filename = os.path.splitext(filename)[0]
            f2 = open(norm_filename+'.txt', 'w')
            z = 0
            for line in f1:
                if line == '<!----------- Собственно произведение --------------->\n':
                    z = 1
                elif z == 1 and line == '<!--------------------------------------------------->\n':
                    z = 0
                elif z == 1:
                    soup = BeautifulSoup(line)
                    f2.write(soup.get_text())
            f1.close()
            f2.close()
            print("IWH")
#f = open('D:\TestSamlib\Абабков Андрей С.. Укус Лунного Вампира_.html', 'r')
#f2 = open('D:\TestSamlib\Абабков Андрей С.. Укус Лунного Вампира_.txt', 'w')
#z = 0
#for line in f:
 #   if line == '<!----------- Собственно произведение --------------->\n':
  #      z = 1
   # elif z == 1 and line == '<!--------------------------------------------------->\n':
    #    z = 0
    #elif z == 1:
    #    soup = BeautifulSoup(line)
    #    f2.write(soup.get_text())
#f.close()
#f2.close()