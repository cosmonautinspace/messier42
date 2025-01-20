import pandas as pd 
fileName = input("Enter the name of the file to preprocess >> ")
data = pd.read_csv(f'data/scrapeFromHere/{fileName}.csv')



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
data = data.drop(columns=['rT','gT','bT'])
data = data.drop_duplicates(keep='first')

data.reset_index(drop=True, inplace=True)

print(data)

data.to_csv(f'data/scrapeFromHere/{fileName}_preproc.csv', sep=',', index=False)

