from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import requests
import functools
import time
import re
import os
from lxml.html import fromstring
from itertools import cycle
class MyException(Exception):
    def __init__(self, message):
        self.message = message
def read_info(info_name):
    #fileinfo = text_number + 'Info.txt'
    res = []
    for line in open(info_name, "r"):
        res.append(line.rstrip())
    return res
def try_multiple_proxy(address, proxies):
    #proxies = {
    #    "http": 'http://209.50.52.162:9050',
    #    "https": 'http://209.50.52.162:9050'
    #}
    header =  {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'
    }
    #header = UserAgent.chrome()

    r = requests.get(address, headers=header, proxies = proxies, timeout=10)
    print(r.status_code)
    if r.status_code != requests.codes.ok:
        print("I tried multiple")
        #time.sleep(1)
        r = requests.get(address, headers=header, proxies = proxies, timeout=10)
        if r.status_code != requests.codes.ok:
            print("I tried multiple2")
            #time.sleep(1)
        r = requests.get(address, headers=header, proxies = proxies, timeout=10)
    return r
def try_multiple(address):
    #proxies = {
    #    "http": 'http://209.50.52.162:9050',
    #    "https": 'http://209.50.52.162:9050'
    #}
    r = requests.get(address, headers={'User-Agent': UserAgent().chrome})
    print(r.status_code)
    if r.status_code != requests.codes.ok:
        print("I tried multiple")
        #time.sleep(1)
        r = requests.get(address, headers={'User-Agent': UserAgent().chrome})
        if r.status_code != requests.codes.ok:
            print("I tried multiple2")
            #time.sleep(1)
        r = requests.get(address, headers={'User-Agent': UserAgent().chrome})
    return r


def get_book(query, proxies):
    search_res = try_multiple_proxy(query, proxies)
    soup = BeautifulSoup(search_res.text, 'html.parser')
    links = soup.findAll('a', href=True)
    print(links)
    allhrefs = list(map(lambda x: x['href'], links))
    #bookhref = list(filter(lambda x: re.search("https://www.livelib.ru/book/*", str(x)) != None, allhrefs))
    bookhref = list(filter(lambda x: re.match("/book/\d.*", str(x)) != None, allhrefs))
    print(bookhref)
    #objectwrapper = soup.find('div', {'class' : "object-wrapper object-edition"})
    #if objectwrapper == None:
    #    return "NoBook"
    #print(objectwrapper['onclick'])
    #bookhref = re.findall("'*'", objectwrapper['onclick'])[0]
    print(bookhref)
    if len(bookhref) < 1:
        if len(allhrefs) < 2:
            raise MyException("Livelib is a domain of very bad ice giants")
        return "NoBook"
    return 'https://www.livelib.ru' + bookhref[0]
def get_proxies():
    url = 'https://www.sslproxies.org/'
    response = requests.get(url)
    parser = fromstring(response.text)
    proxies = set()
    for i in parser.xpath('//tbody/tr')[:10]:
        if i.xpath('.//td[7][contains(text(),"yes")]'):
            #Grabbing IP and corresponding PORT
            proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
            proxies.add(proxy)
    return proxies
def proxies_for_duckduckgo(query):
    proxies = get_proxies()
    proxy_pool = cycle(proxies)
    url = query
    i = 0
    while (i < 40):
        # Get a proxy from the pool
        proxy = next(proxy_pool)
        print("Request #%d" % i)
        try:
            retbook = get_book(query, {"http": proxy, "https": proxy})
            return retbook
        except MyException:
            print("Ice giants have broken pact again")
        except:
            # Most free proxies will often get connection errors. You will have retry the entire request using another proxy to work.
            # We will just skip retries as its beyond the scope of this tutorial and we are only downloading a single url
            print("Skipping. Connnection error")
        i += 1
    return "NoBook"
def proxies_for_livelib(query):
    proxies = get_proxies()
    proxy_pool = cycle(proxies)
    i = 0
    while (i < 40):
        # Get a proxy from the pool
        proxy = next(proxy_pool)
        print("Request #%d" % i)
        try:
            rettags = get_tags(query, {"http": proxy, "https": proxy})
            return rettags
        except MyException:
            print("Ice giants have broken pact again")
        except:
            # Most free proxies will often get connection errors. You will have retry the entire request using another proxy to work.
            # We will just skip retries as its beyond the scope of this tutorial and we are only downloading a single url
            print("Skipping. Connnection error")
        i += 1
    return "NoTags"
def get_tags(book, proxies):
    #search_res = try_multiple(book)
    search_res = try_multiple_proxy(book, proxies)
    search_res.encoding = 'utf-8'
    soup = BeautifulSoup(search_res.text, 'html.parser')
    links = soup.findAll('a', href=True)
    print(links)
    taghref = list(filter(lambda x: re.search("https://www.livelib.ru/tag/*", str(x['href'])) != None, links))
    tags = list(map(lambda x: x.text, taghref))
    if(len(tags) < 1 and len(links) < 2):
        raise MyException("Livelib is the domain of ice giants")
    return tags
def write_tags(text_number):
    if(not os.path.isfile(text_number+'Tags.txt')):
        filetag = text_number + 'Tags.txt'
        try:
            text_info = read_info(text_number + 'Info.txt')
        except FileNotFoundError:
            print("File not found")
            return
        if(len(text_info) < 3):
            print("Problem with info!")
            return
        print(text_info)
        try:
            #query = 'http://duckduckgo.com/html/?q=' + text_info[0] + '+' + text_info[
            #    1] + '+рецензии+на+книгу+' + functools.reduce(lambda s1, s2: s1 + '+' + s2,
            #                                                  text_info[2].split(' ')) + '+site%3Alivelib.ru&t=h_&ia=web'
            #book = proxies_for_duckduckgo(query)
            query = 'https://www.livelib.ru/find/books/'+text_info[0] + '+' + text_info[1] + '+' + functools.reduce(lambda s1, s2: s1 + '+' + s2,  text_info[2].split(' '))
            print(query)
            #book = proxies_for_duckduckgo(query)
            #book = get_book(query, None)
            if(not os.path.isfile(text_number+'TextName.txt')):
                book = proxies_for_duckduckgo(query)
                if book != "NoBook":
                    newf = open(text_number + 'TextName.txt', "w")
                    newf.write(book)
                    newf.close()
            else:
                newf = open(text_number+'TextName.txt', "r")
                book = newf.read()
                newf.close()
            print(book)
            if book == "NoBook":
                print("Book isn't found, tags aren't created")
                return
            book = book + '?PageSpeed=noscript'
            print(book)
            tags = proxies_for_livelib(book)
            print(tags)
            f = open(filetag, "w")
            if tags:
                print(functools.reduce(lambda x,y: str(x) + '\n' + str(y),tags))
                f.write(functools.reduce(lambda x,y: str(x) + '\n' + str(y), tags))
            else:
                print("All is bad")
            f.close()
        except ZeroDivisionError:
            print("What")
            0/0

#print(proxies_for_livelib('https://www.livelib.ru/book/1001296737-anastasiya-zagadka-velikoj-knyazhny-piter-kurt?PageSpeed=noscript'))
#0/0
#text_number = 'D:\TestFlibusta\9373Info.txt'
#query = "https://www.livelib.ru/find/books/%D0%95%D0%BB%D0%B8%D0%B7%D0%B0%D0%B2%D0%B5%D1%82%D0%B0+%D0%A8%D1%83%D0%BC%D1%81%D0%BA%D0%B0%D1%8F+%D0%A6%D0%B5%D0%BD%D0%B0+%D1%81%D0%BB%D0%BE%D0%B2%D0%B0"
#query = "https://www.livelib.ru/find/books/Николай+Носов+Огурцы"
#book = proxies_for_duckduckgo(query)
#print(book)
#0/0
#book = get_book(query)
#fb2book = 'D:\TestFlibusta\\110284'
#write_tags(fb2book)
processed = 0
num_errors = 0
MAX_PROCESSED = 120000
for file in os.listdir("D:\TestFlibusta"):
    try:
        if file.endswith(".fb2"):
            #filename = 'D:\TestFlibusta\9373.fb2'
            if processed > 9320:
                filename = 'D:\TestFlibusta\\' + file
                text_number = filename.split('.')[0]
                print(text_number)
                write_tags(text_number)
            processed += 1
        if processed >= MAX_PROCESSED:
            break
    #except TypeError:
    #    print("Ok, some type error, just ignore it")
     #   num_errors += 1
    #    processed += 1
    except ValueError:
        num_errors += 1
        processed += 1
print(num_errors)