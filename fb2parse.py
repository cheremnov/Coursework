import functools
import os
class MyException(Exception):
    def __init__(self, message):
        self.message = message
def parse_author(e):
    author_name = e.find('.//{http://www.gribuser.ru/xml/fictionbook/2.0}first-name')
    author_surname = e.find('.//{http://www.gribuser.ru/xml/fictionbook/2.0}last-name')
    if author_name == None or author_surname == None:
        raise MyException("None")
    return author_name.text + '\n' + author_surname.text
def parse_title(e):
    title = e.find('.//{http://www.gribuser.ru/xml/fictionbook/2.0}book-title')
    if title == None:
        raise MyException("None")
    return title.text
def write_info(text_number, e):
    fileinfo = text_number + 'Info.txt'
    if (not os.path.isfile(fileinfo)):
        f = open(fileinfo, "w")
        infotext = parse_author(e) + '\n' + parse_title(e)
        f.write(infotext)
        f.close()


import xml.etree.ElementTree
processed = 0
num_errors = 0
MAX_PROCESSED = 20000
for file in os.listdir("D:\TestFlibusta"):
    try:
        if file.endswith(".fb2"):
            #filename = 'D:\TestFlibusta\9373.fb2'
            filename = 'D:\TestFlibusta\\' + file
            text_number = filename.split('.')[0]
            e = xml.etree.ElementTree.parse(filename)
            write_info(text_number, e)
            #print(text_number)
            processed += 1
        if processed >= MAX_PROCESSED:
            break
    except TypeError:
        print("Ok, some type error, just ignore it")
        num_errors += 1
    except UnicodeEncodeError:
        print("Ok, some unicode error, just ignore it")
        num_errors += 1
    except xml.etree.ElementTree.ParseError:
        print("Ok, parse error, ignore it again")
        num_errors += 1
    except MyException:
        print("Hooray, my exception")
        num_errors += 1
print(num_errors)
#print(root[0][0][1][0].text + '\n' + root[0][0][1][1].text)
#print(e.text)
#for atype in e.findall('first-name'):
#    print(atype.text)
#from requests_xml import XMLSession
#session = XMLSession()


#r = session.get('https://www.nasa.gov/rss/dyn/lg_image_of_the_day.rss')
#item = r.xml.xpath('//first-name', first=True)
#print(item.text)