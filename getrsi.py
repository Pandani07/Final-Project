from lxml import html
import requests
import json
import argparse
from collections import OrderedDict
from bs4 import BeautifulSoup

url = "https://www.moneycontrol.com/technical-analysis/indian-indices/nifty-50-9"

def getrsi():
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    rsi = soup.find_all('div', {'class':'mtindi FR'})[0].find('div', {'class':'mt20'}).find_all('td')[1].text
    return rsi

