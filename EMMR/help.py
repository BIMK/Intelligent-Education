import pandas as pd

def split(x):
    new = ''
    for i in x:
        for j in i:
            new = new + j
div_list = pd.read_csv('util_data/' + 'Music' + '_div.csv', sep=',', header=None, usecols=[0, 1], names=["item", "genre"])

split(div_list['genre'][0])

