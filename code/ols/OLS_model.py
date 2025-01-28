import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv('data/cleanedData/placeholderData_trainData.csv')

gMMax = data["gM"].max()
rbMMax = data["rbM"].max()
data["gM"] = data["gM"] / gMMax
data["rbM"] = data["rbM"] / rbMMax

X_rb = data[['rI', 'bI']]
y_rb = data['rbM']
X_g = data[['gI']]
y_g = data['gM']

X_rb_train, X_rb_test, y_rb_train, y_rb_test = train_test_split(X_rb, y_rb, test_size=0.2, random_state=44)
X_g_train, X_g_test, y_g_train, y_g_test = train_test_split(X_g, y_g, test_size=0.2, random_state=44)

X_rb_train_sm = sm.add_constant(X_rb_train)
ols_rb_model = sm.OLS(y_rb_train, X_rb_train_sm).fit()
X_rb_test_sm = sm.add_constant(X_rb_test)
y_rb_pred = ols_rb_model.predict(X_rb_test_sm)

X_g_train_sm = sm.add_constant(X_g_train)
ols_g_model = sm.OLS(y_g_train, X_g_train_sm).fit()
X_g_test_sm = sm.add_constant(X_g_test)
y_g_pred = ols_g_model.predict(X_g_test_sm)

with open("code/ols/currentOlsSolution_RB.pkl", "wb") as f:
    pickle.dump(ols_rb_model, f)
with open("code/ols/currentOlsSolution_G.pkl", "wb") as f:
    pickle.dump(ols_g_model, f)

with open("code/ols/currentOlsSolution_RB.txt", "w") as f:
    f.write(ols_rb_model.summary().as_text())
with open("code/ols/currentOlsSolution_G.txt", "w") as f:
    f.write(ols_g_model.summary().as_text())

np.savetxt("code/ols/rb_ols_residuals.csv", ols_rb_model.resid, delimiter=",")
np.savetxt("code/ols/g_ols_residuals.csv", ols_g_model.resid, delimiter=",")

print("OLS models saved.")
