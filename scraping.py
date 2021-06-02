from lxml import html
import requests
import json
import argparse
from collections import OrderedDict
from bs4 import BeautifulSoup

nse_url = "https://www.moneycontrol.com/technical-analysis/indian-indices/nifty-50-9"
infy_url ='https://www.moneycontrol.com/technical-analysis/infosys/IT'
techm_url = 'https://www.moneycontrol.com/technical-analysis/techmahindra/TM4'
wipro_url = 'https://www.moneycontrol.com/technical-analysis/wipro/W'
adani_url = 'https://www.moneycontrol.com/technical-analysis/AdaniPorts/MPS'
cipla_url = 'https://www.moneycontrol.com/technical-analysis/cipla/C'
axis_bank_url = 'https://www.moneycontrol.com/technical-analysis/axisbank/AB16'
hcl_url = 'https://www.moneycontrol.com/technical-analysis/hcltechnologies/HCL02'
hdfc_url = 'https://www.moneycontrol.com/technical-analysis/hdfcbank/HDF01'
jswsteel_url = 'https://www.moneycontrol.com/technical-analysis/jswsteel/JSW01'
reliance_url = 'https://www.moneycontrol.com/technical-analysis/relianceindustries/RI'
upl_url = 'https://www.moneycontrol.com/technical-analysis/upl/UP04'
ongc_url= 'https://www.moneycontrol.com/technical-analysis/oilnaturalgascorporation/ONG'
hindun_url = 'https://www.moneycontrol.com/technical-analysis/hindustanunilever/HU'
itc_url = 'https://www.moneycontrol.com/technical-analysis/itc/ITC'
tcs_url = 'https://www.moneycontrol.com/technical-analysis/tataconsultancyservices/TCS'

company_dict = {
    'NIFTY 50': nse_url,
    'Adani': adani_url,
    'Axis': axis_bank_url,
    'Cipla': cipla_url,
    'HCL':hcl_url,
    'HDFC Bank':hdfc_url,
    'Hindustan Unilever': hindun_url,
    'Infosys': infy_url,
    'ITC': itc_url,
    'JSW Steel': jswsteel_url,
    'ONGC': ongc_url,
    'Reliance': reliance_url,
    'TATA Consultancy Services': tcs_url,
    'Tech Mahindra': techm_url,
    'UPL': upl_url,
    'Wipro': wipro_url,
}


def get_rsi(company):

    url = company_dict.get(company)
    print("Key: {}, Company: {}".format(company, url))
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    rsi = soup.find_all('div', {'class':'mt20'})[3].find_all('td')[1].text
    return rsi

def get_nse_rsi():
    nse_url = "https://www.moneycontrol.com/technical-analysis/indian-indices/nifty-50-9"
    response = requests.get(nse_url)
    soup = BeautifulSoup(response.text, 'lxml')
    rsi = soup.find_all('div', {'class':'mtindi FR'})[0].find('div', {'class':'mt20'}).find_all('td')[1].text
    return rsi