import pandas as pd 

data = pd.read_csv('data/scrapeFromHere/pixelValues.csv')



#Since CMOS sensors employ the Bayer pattern, which has a 1:2:1 ratio of RGB subpixels, 
#multiplication factor for Green needs to be calculated seperately.
#Also, this theoretically helps with green noise removal.

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
data = data.drop(columns=['rT','gT','bT'])q
data = data.drop_duplicates()
'''
#normalization
#The maximum possible theoretical multiplaction factor for a pixel is 255 (when value is amplified from 1 to 255)
#Also the maximum possible value for a pixel is 255, so both the raw pixel values and the MFactor as divided by 255 for normalization
'''

'''
for i in range(data['gM'].size):
    data['gM'].iloc[[i]] /= 255
    data['rbM'].iloc[[i]] /= 255
    data['gI'].iloc[[i]] /= 255
    data['rI'].iloc[[i]] /= 255
    data['bI'].iloc[[i]] /= 255
'''
data.reset_index(drop=True, inplace=True)

###############Outlier removal###############
#quartile filter

#zscore filter

print(data)

data.to_csv('test.csv', sep=',', index=False)