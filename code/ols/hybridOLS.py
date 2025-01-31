import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import pickle
from UE_04_LinearRegDiagnostic import LinearRegDiagnostic

trainData = pd.read_csv('data/cleanedData/train_data.csv')
testData = pd.read_csv('data/cleanedData/test_data.csv')

X_train = trainData[['rI','gI','bI']]
y_train = trainData[['gM','rbM']]

X_test = testData[['rI','gI','bI']]
y_test = testData[['gM','rbM']]

X_train_sm = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_sm).fit()
X_test_sm = sm.add_constant(X_test)
y_pred = ols_model.predict(X_test_sm)

print(ols_model.predict(X_test_sm))

with open("code/ols/currentOlsSolution.pkl", "wb") as f:
    pickle.dump(ols_model, f)

print(ols_model.summary())

np.savetxt("documentation/visualizations/source/ols/ols_residuals.csv", ols_model.resid, delimiter=",")


print("OLS model saved.")
