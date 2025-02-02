import statsmodels.api as sm
import pandas as pd

activationdf = pd.read_csv(f'/tmp/ai_system/activationBase/activation_data.csv')

x_rb = activationdf[['rI', 'bI']]
x_rb_sm= sm.add_constant(x_rb, has_constant='add')
x_g = activationdf['gI']
x_g_sm = sm.add_constant(x_g, has_constant='add')

olsG = sm.load('/tmp/ai_system/knowledgeBase/ols_model/currentOlsSolution_G.pkl')
olsRB = sm.load('/tmp/ai_system/knowledgeBase/ols_model/currentOlsSolution_RB.pkl')
rbM = olsRB.predict(x_rb_sm)
gM = olsG.predict(x_g_sm)
print(f"The OLS activation for the given input csv file is (mean normalized) >> rbM = {rbM} , gM = {gM}]")
