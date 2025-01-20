import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''Some test visualisations nothing important as of yet'''

df = pd.read_csv('data/extras/orion.tif_16bitvalues_preproc_cleanedDataWhole.csv')

#plt.scatter(df['gM'], df['gI'], c='green')
plt.scatter(df['rbM'], df['rI'], c='red')
#plt.scatter(df['rbM'], df['bI'], c='blue')


plt.show()