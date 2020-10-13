import pip
pip.main(['install','logsumexp'])

import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm 
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.varmax import VARMAX
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()



def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window = 12).mean()
    rolstd = timeseries.rolling(window = 12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color = 'blue',label = 'Original')
    mean = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
    std = plt.plot(rolstd, color = 'black', label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block = False)
    plt.pause(30)

data = pd.read_csv('/home/u73/workspace/dataset.csv').dropna()
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace = True)

columns = ['month', 'Weekday', 'WeekOfMonth', 'isWeekend', 'Holiday','islongweekend']
for column in columns:
	data[column] = data[column].astype('int64')
print(data.isnull().any())
print(data.columns)

mean = data['qty'].rolling(window = 12).mean()
data['qty'].plot()
mean.plot(color = 'red')
plt.show()

############################----removing outliers-------#########################################
min_quantile = data['qty'].quantile(0.225)
max_quantile = data['qty'].quantile(0.975)
data['qty'] = data['qty'].mask(data['qty'] > max_quantile, max_quantile)
data['qty'] = data['qty'].mask(data['qty'] < min_quantile, min_quantile)
mean = data['qty'].rolling(window = 7).mean()
data['qty'].plot()
mean.plot(color = 'red')
plt.show()
###########################--------Stats Tests-----------########################################
print('Performing adfuller test....')
x = data.iloc[:,6]
result = adfuller(x)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
if result[1] < 0.05: 
	print('Not stationary')
else: 
	print('stationary')
print('\n')
x.plot()
plt.show()

print('After log transformation')
print('Performing adfuller test......')
data['qty'] = np.log(data['qty'])
x = data.iloc[:,6]
result = adfuller(x)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
if result[1] < 0.05: 
	print('Not stationary')
else: 
	print('stationary')
x.plot()
plt.show()
plt.pause(5)

test_stationarity(data['qty'])

data['qty'] = pd.Series(np.log(data['qty']).diff().dropna())
data.dropna(inplace = True)
# data.plot()
# plt.show()


# train, validate = train_test_split(data, test_size = 0.3)
train = data[:int(0.8*(len(data)))]
validate = data[int(0.2*(len(data))):]

model = VARMAX(endog = train, enforce_stationarity = True)
model_fit = model.fit(maxiters = 1)
print('-----------RESULTS----------------')
print(model_fit.summary())
prediction = model_fit.predict(start = datetime.strptime('20180101','%Y%m%'),steps=len(validate))
print(prediction)
print('Variables for th model %s' %result.exog_names)
order = result.k_ar
forecast_values = pd.DataFrame(data = result.forecast(y = data['qty'].values,steps = 5))
result.plot_forecast(steps = 5, plot_stderr = False)

pred = model_fit.forecast(model_fit.y, steps = len(validate))

