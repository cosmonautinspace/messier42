import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import pickle
from UE_04_LinearRegDiagnostic import LinearRegDiagnostic
import matplotlib.pyplot as plt 

data = pd.read_csv('/tmp/ai_system/learningBase/train/train_data.csv')
testdf = pd.read_csv('/tmp/ai_system/learningBase/validation/test_data.csv')


X_rb_train = data[['rI', 'bI']]
X_rb_test = testdf[['rI', 'bI']] 
y_rb_train = data['rbM']
y_rb_test = testdf['rbM']

X_g_train = data['gI'] 
X_g_test = testdf['gI']
y_g_train = data['gM']
y_g_test = testdf['gM']

X_rb_train_sm = sm.add_constant(X_rb_train)
ols_rb_model = sm.OLS(y_rb_train, X_rb_train_sm).fit()
X_rb_test_sm = sm.add_constant(X_rb_test)
y_rb_pred = ols_rb_model.predict(X_rb_test_sm)

X_g_train_sm = sm.add_constant(X_g_train)
ols_g_model = sm.OLS(y_g_train, X_g_train_sm).fit()
X_g_test_sm = sm.add_constant(X_g_test)
y_g_pred = ols_g_model.predict(X_g_test_sm)

with open("/tmp/ai_system/currentOlsSolution_RB.pkl", "wb") as f:
    pickle.dump(ols_rb_model, f)
with open("/tmp/ai_system/currentOlsSolution_G.pkl", "wb") as f:
    pickle.dump(ols_g_model, f)

with open("/tmp/ai_system/currentOlsSolution_RB.txt", "w") as f:
    f.write(ols_rb_model.summary().as_text())
with open("/tmp/ai_system/currentOlsSolution_G.txt", "w") as f:
    f.write(ols_g_model.summary().as_text())


cls = LinearRegDiagnostic(ols_g_model)
vif, fig, ax= cls()
print(vif)

fig.savefig('/tmp/ai_system/GreenChannelDiagnosticPlots.pdf', format="pdf")


cls = LinearRegDiagnostic(ols_rb_model)
vif, fig, ax= cls()
print(vif)

fig.savefig('/tmp/ai_system/Red&BlueChannelDiagnosticPlots.pdf', format="pdf")

print("OLS models saved.")

predictedG = ols_g_model.predict(X_g_test_sm)
predictedRB = ols_rb_model.predict(X_rb_test_sm)

plt.scatter(testdf['gI'], testdf['gM'], c='green',s=1)
plt.scatter(testdf['gI'], predictedG, c='orange', s=1)
plt.xlabel('Green sub pixel values (Mean normalized)')
plt.ylabel('Multiplication Factor (Mean normalized)')
plt.legend(['Actual', 'Predicted'])
plt.title('OLS Model Performance - Green Channel')
plt.savefig('/tmp/ai_system/OLS_green_performance.png')
plt.close()

plt.scatter(testdf['rI'], testdf['rbM'], c='red',s=1)
plt.scatter(testdf['rI'], predictedRB, c='orange', s=1)
plt.xlabel('Red sub pixel values (Mean normalized)')
plt.ylabel('Multiplication Factor (Mean normalized)')
plt.legend(['Actual', 'Predicted'])
plt.title('OLS Model Performance - Red Channel')
plt.savefig('/tmp/ai_system/OLS_red_performance.png')
plt.close()

plt.scatter(testdf['bI'], testdf['rbM'], c='blue',s=1)
plt.scatter(testdf['rI'], predictedG, c='orange', s=1)
plt.xlabel('Blue sub pixel values (Mean normalized)')
plt.ylabel('Multiplication Factor (Mean normalized)')
plt.legend(['Actual', 'Predicted'])
plt.title('OLS Model Performance - Blue Channel')
plt.savefig('/tmp/ai_system/OLS_blue_performance.png')
plt.close()
