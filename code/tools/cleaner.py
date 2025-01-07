import pandas as pd 

data = pd.read_csv('code/tools/input2.csv')



#Since CMOS sensors use Bayer pattern which has a 1:2:1 ratio of RGB subpixels, 
#multiplication factor for Green needs to be calculated seperately.
#Also, this theoretically helps with green noise removal.

#red and blue subpixel multiplication factor
rbMfactor = []
#green subpixel multiplication factor
gMfactor = []

for index, row in data.iterrows():
    rFactor = int(row['rT']/row['rI'])
    bFactor = int(row['bT']/row['bI'])
    gFactor = int(row['gT']/row['gI'])
    averageBR = int((rFactor+bFactor)/2)
    rbMfactor.append(averageBR)
    gMfactor.append(gFactor)

data.insert(loc=0, column='rbM', value=rbMfactor)
data.insert(loc=0, column='gM', value=gMfactor)
data = data.drop(columns=['rT','gT','bT'])
data = data.drop_duplicates()
print(data)
data.to_csv('test.csv', sep=',', index=False)