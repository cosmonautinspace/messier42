import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

test = pd.read_csv('data/cleanedData/unnormalized_test_data.csv')
train = pd.read_csv('data/cleanedData/unnormalized_train_data.csv')


plt.scatter(test['gI'], test['gM'])
plt.show()