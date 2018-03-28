import scipy as sy
import pandas as pd
import numpy as np
import math

def scaleOutMarketCap (df, features):
    for feature in features:
        df.loc[:,feature] = df.loc[:,feature]/df.loc[:,'Market_cap'] * 10000
    return df

def encode_sector (df):
    sector = df ['Sector']
    encoding = []
    for index, row in sector.iteritems():
        if (row == 'Technology'):
            tup = (index, 1)
            encoding.append(tup)
        if (row == 'Industrial'):
            encoding.append((index, 2))
        if (row == 'Communications'):
            encoding.append((index, 3))
        if (row == 'Financial'):
            encoding.append((index, 4))
        if (row == 'Basic Materials'):
            encoding.append((index, 5))
        if (row == 'Consumer, Non-cyclical'):
            encoding.append((index, 6))
        if (row == 'Consumer, Cyclical'):
            encoding.append((index, 7))
        if (row == 'Utilities'):
            encoding.append((index, 8))
        if (row == 'Energy'):
            encoding.append((index, 9))
    encoding = pd.DataFrame(encoding, columns=['Security', 'Sector'])
    encoding = encoding.set_index('Security')
    return encoding

def encode_binary (df):
    bribery = df['Bribery']
    ethics = df['Ethics']
    bribery_l = []
    for index, row in bribery.iteritems():
        if (row == 'N'):
            bribery_l.append((index, 0))
        if (row == 'Y'):
            bribery_l.append((index, 1))
    ethics_l = []
    for index, row in ethics.iteritems():
        if (row == 'N'):
            ethics_l.append((index, 0))
        if (row == 'Y'):
            ethics_l.append((index, 1))
    bribery_l = pd.DataFrame(bribery_l, columns=['Security', 'Bribery'])
    bribery_l = bribery_l.set_index('Security')
    ethics_l = pd.DataFrame(ethics_l, columns=['Security', 'Ethics'])
    ethics_l = ethics_l.set_index('Security')
    
    return bribery_l, ethics_l

def encodeMarketCap(df):
    size = df['Market_cap'] * 1000000
    encoded_size = []
    for index, row in size.iteritems():
        if (10000000000 <= row ):
            encoded_size.append((index, 4)) #'large cap'
        elif (2000000000 <= row and row < 10000000000):
            encoded_size.append((index, 3)) #'mid cap'
        elif (300000000 <= row and row < 2000000000):
             encoded_size.append((index, 2)) #'small cap'
        elif (50000000 <= row and row < 300000000):
             encoded_size.append((index, 1)) #'micro cap'
        elif (row < 50000000):
            encoded_size.append((index, 0))  #'nano cap'
        else: 
            encoded_size.append((index, math.nan))
    encoded_size = pd.DataFrame(encoded_size, columns=['Security', 'Size'])
    encoded_size = encoded_size.set_index('Security')
    return encoded_size

