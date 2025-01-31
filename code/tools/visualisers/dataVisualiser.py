import matplotlib.pyplot as plt
import pandas as pd

testdf = pd.read_csv('data/cleanedData/test_data.csv')
traindf = pd.read_csv('data/cleanedData/train_data.csv')

plt.boxplot([traindf['rI'], traindf['gI'], traindf['bI'],
            traindf['rbM'], traindf['gM']], 
            showfliers=True,showcaps=True,showmeans=True, labels=["R","G","B", "RB MFactors", "G Mfactor"])
plt.title('Box plot of training data (mean normalized)')
plt.savefig('documentation/visualizations/TrainingBoxPlots.png')
plt.close()

plt.boxplot([testdf['rI'], testdf['gI'], testdf['bI'],
            testdf['rbM'], testdf['gM']], 
            showfliers=True,showcaps=True,showmeans=True, labels=["R","G","B", "RB MFactors", "G Mfactor"])
plt.title('Box plot of testing data (mean normalized)')
plt.savefig('documentation/visualizations/TestingBoxPlots.png')
plt.close()