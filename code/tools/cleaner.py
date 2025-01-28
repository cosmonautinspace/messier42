import pandas as pd 
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
import random

fileName = input("Enter the name of the file to clean >> ")
data = pd.read_csv(f'data/scrapedData/{fileName}.csv')

'''Normalization Pass 1 (only normalizing subpixel values here, multiplication factor values get normalized at the end)
subpixel values get divided by 255 as it is the max value a pixel can take for an image with a depth of 8 bits
'''

data['gI'] /= 255
data['rI'] /= 255
data['bI'] /= 255
data['rT'] /= 255
data['gT'] /= 255
data['bT'] /= 255

print("Normalization pass 1 complete")

'''Calculating Multiplication factor
Since CMOS sensors employ the Bayer pattern, which has a 1:2:1 ratio of RGB subpixels, 
multiplication factor for Green needs to be calculated seperately.
Also, this theoretically helps with green noise removal. ''' 

#red and blue subpixel multiplication factor
rbMfactor = []
#green subpixel multiplication factor
gMfactor = []

#calculating the multiplication factors for
for index, row in data.iterrows():
    rFactor = int(row['rT']/row['rI'])
    bFactor = int(row['bT']/row['bI'])
    gFactor = int(row['gT']/row['gI'])
    averageBR = int((rFactor+bFactor)/2)
    rbMfactor.append(averageBR)
    gMfactor.append(gFactor)

data.insert(loc=0, column='rbM', value=rbMfactor)
data.insert(loc=0, column='gM', value=gMfactor)

#The raw target pixel values are dropped as they are not needed once the multiplication factor has been calculated
data = data.drop(columns=['rT','gT','bT'])
data = data.drop_duplicates(keep='first')

data.reset_index(drop=True, inplace=True)

print('multiplication factors computed')

print(data)

data.to_csv(f'data/extras/{fileName}_preproc.csv', sep=',', index=False)


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


'''Normalizing Multiplication factor values'''
'''Multiplication factors are normalized when the data is loaded so that the normalization factor is stored in memory
the code for this can be found below but is comented out'''

'''
gMMax = data["gM"].max()
rbMMax = data["rbM"].max()
for i in range(data["gM"].size):
    data["gM"].iloc[[i]] /= gMMax

for i in range(data["rbM"].size):
    data["rbM"].iloc[[i]] /= rbMMax
'''
    
train, test = train_test_split(data, test_size=0.2, train_size=0.8)

activationData = data.iloc[[f'{random.randint(0,10)}']] 

train.to_csv(f'data/cleanedData/train_data.csv', index=False)
test.to_csv(f'data/cleanedData/test_data.csv', index=False)
data.to_csv(f'data/extras/joint_datacollection.csv', index=False)
activationData.to_csv(f'data/cleanedData/activation_data.csv', index=False)