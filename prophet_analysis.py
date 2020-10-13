import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('/home/u73/workspace/dataset.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.rename(columns = {'date':'ds','qty':'y'})
rows = int(data.shape[0]*0.8)
train_data = data.iloc[:rows,:]
test_data = data.iloc[rows:,:]

model = Prophet()
for i in range(5):
	model.fit(train_data)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(test_data)
pred = forecast.yhat
print(np.sqrt(mean_squared_error(test_data['y'],pred)))
# test_data['y'].plot()
# pred.plot()
# plt.show()