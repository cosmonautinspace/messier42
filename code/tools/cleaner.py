import pandas as pd 
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 

data = pd.read_csv('data/scrapedData/placeholderData.csv')

# Basic Data cleaning
data["gM"] = pd.to_numeric(data["gM"],errors="coerce")
data["rbM"] = pd.to_numeric(data["rbM"],errors="coerce")
data["gI"] = pd.to_numeric(data["gI"],errors="coerce")
data["rI"] = pd.to_numeric(data["rI"],errors="coerce")
data["bI"] = pd.to_numeric(data["bI"],errors="coerce")


data.drop_duplicates(keep='first')
data.reset_index(drop=True, inplace=True)
data.dropna()

'''Quartile filter
Using quartile filter before z score filter to remove really extreme values that might "bias" the z score filter
'''

q95 = data.quantile(0.95)
q5 = data.quantile(0.05)

#removal of outliers with odd multiplication factor values
to_drop = []
for i in range(data["gM"].size):
    if (data["gM"].iloc[i] > q95.gM) | (data["gM"].iloc[i] < q5.gM):
        to_drop.append(i)
    if (data["rbM"].iloc[i] > q95.rbM) | (data["rbM"].iloc[i] < q5.rbM):
        to_drop.append(i)
data.drop(to_drop, inplace=True)
data.reset_index(drop=True, inplace=True)

#removal of outliers with odd pixel values (dead pixels, extreme noise, etc.)
to_drop = []
for i in range(data["rI"].size):
    if (data["rI"].iloc[i] > q95.rI) | (data["rI"].iloc[i] < q5.rI):
        to_drop.append(i)
    if (data["gI"].iloc[i] > q95.gI) | (data["gI"].iloc[i] < q5.gI):
        to_drop.append(i)
    if (data["bI"].iloc[i] > q95.bI) | (data["bI"].iloc[i] < q5.bI):
        to_drop.append(i)
data.drop(to_drop, inplace=True)
data.reset_index(drop=True, inplace=True)

'''Z score filter'''
zscore_threshold = 1.65

zscoregM = zscore(data["gM"])

to_drop = []
for i in range(0,zscoregM.size):
    if abs(zscoregM[i])>zscore_threshold:
        to_drop.append(i)

data.drop(to_drop,inplace=True)
data.reset_index(drop=True, inplace=True)

to_drop = []
zscorerbM = zscore(data["rbM"])
for i in range(0,zscorerbM.size):
    if abs(zscorerbM[i])>zscore_threshold:
        to_drop.append(i)
data.drop(to_drop,inplace=True)
data.reset_index(drop=True, inplace=True)

'''Normalization
Add normalization code later if the chosen model benefits from it.
'''

train, test = train_test_split(data, test_size=0.2, train_size=0.8)

train.to_csv('data/cleanedData/trainData.csv', index=False)
test.to_csv('data/cleanedData/testData.csv', index=False)

