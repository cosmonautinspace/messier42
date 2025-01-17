import pandas as pd
from sklearn.model_selection import train_test_split

csvName = input('Enter the name of the CSV to split (will be split 4 ways)')

primaryDf = pd.read_csv(f"data/scrapeFromHere/{csvName}")

firstDf, secondDf = train_test_split(primaryDf, test_size=0.5)

firstDf, fourthDf = train_test_split(firstDf,test_size=0.5)
secondDf, thirdDf = train_test_split(secondDf,test_size=0.5)

firstDf.to_csv(f"data/scrapeFromHere/first_{csvName}", index=False)
secondDf.to_csv(f"data/scrapeFromHere/second_{csvName}", index=False)
thirdDf.to_csv(f"data/scrapeFromHere/third_{csvName}", index=False)
fourthDf.to_csv(f"data/scrapeFromHere/fourth_{csvName}", index=False)


