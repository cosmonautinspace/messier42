import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pybrain.tools.customxml.networkreader import NetworkReader
import statsmodels.api as sm


test = pd.read_csv('data/cleanedData/test_data.csv')

ann = NetworkReader.readFrom('code/ann_PyBrain/currentSolution.xml')

predictedRB = []
predictedG = []

for index, row in test.iterrows():
    predicted = ann.activate([row['rI'],row['gI'],row['bI']])
    predictedRB.append(predicted[1])
    predictedG.append(predicted[0])

plt.scatter(test['gI'], test['gM'], c='green',s=1)
plt.scatter(test['gI'], predictedG, c='orange', s=1)
plt.xlabel('Green sub pixel values (Mean normalized)')
plt.ylabel('Multiplication Factor (Mean normalized)')
plt.legend(['Actual', 'Predicted'])
plt.title('ANN Model Performance - Green Channel')
plt.savefig('documentation/visualisations/ANN_green_performance.png')
plt.close()

plt.scatter(test['rI'], test['rbM'], c='red',s=1)
plt.scatter(test['rI'], predictedRB, c='orange', s=1)
plt.xlabel('Red sub pixel values (Mean normalized)')
plt.ylabel('Multiplication Factor (Mean normalized)')
plt.legend(['Actual', 'Predicted'])
plt.title('ANN Model Performance - Red Channel')
plt.savefig('documentation/visualisations/ANN_red_performance.png')
plt.close()

plt.scatter(test['bI'], test['rbM'], c='blue',s=1)
plt.scatter(test['bI'], predictedRB, c='orange', s=1)
plt.xlabel('Blue sub pixel values (Mean normalized)')
plt.ylabel('Multiplication Factor (Mean normalized)')
plt.legend(['Actual', 'Predicted'])
plt.title('ANN Model Performance - Blue')
plt.savefig('documentation/visualisations/ANN_blue_performance.png')
plt.close()