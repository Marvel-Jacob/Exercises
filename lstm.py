# import pip
# pip.main(['install','matplotlib'])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
from keras.regularizers import l2

# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def logger(var):
	return np.log(var)
def unlogger(var):
	return np.exp(var)
def series_to_supervised(data, n_in = 1, n_out = 1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis = 1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace = True)
	return agg


################# Read the data ##################
dataset = pd.read_csv('/home/u73/workspace/dataset.csv')
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset.set_index('Date', inplace = True)

################## Masking outliers ####################
# min_quantile = dataset['qty'].quantile(0.225)
# max_quantile = dataset['qty'].quantile(0.975)
# dataset['qty'] = dataset['qty'].mask(dataset['qty'] > max_quantile, max_quantile)
# dataset['qty'] = dataset['qty'].mask(dataset['qty'] < min_quantile, min_quantile)

print('\nrange of qty is min: {} to max: {}\n'.format(min(dataset['qty']),max(dataset['qty'])))

dataset_values = dataset.values
dataset_values = dataset_values.astype('float32')
sc = MinMaxScaler(feature_range = (0,1))
scaled_data = sc.fit_transform(dataset_values)
# qty = sc.fit_transform(np.reshape(np.array(dataset['qty']),(-1,1)))
# dataset['qty'] = qty
# dataset.to_csv('qty_transformed.csv', index = False)

# # Feature Scaling
# sc = MinMaxScaler(feature_range = (0,1))
# training_scaled = sc.fit_transform(training_set)

# training_set = series_to_supervised(scaled_data)
# training_set = training_set.drop(training_set.columns[6:], axis = 1)

train = scaled_data[:,0:5]
labels = scaled_data[:,5]

train_x, validate_x, train_y, validate_y = train_test_split(train, labels, test_size = 0.1, shuffle = False)
# actual_qty = validate_y.values
dummies_train_x = train_x
dummy_validate_x = validate_x
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
validate_x = np.reshape(validate_x, (validate_x.shape[0], 1, validate_x.shape[1]))

# Initialisin an RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (train_x.shape[1], train_x.shape[2]), kernel_regularizer = l2(0.001)))
regressor.add(Dropout(0.1))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True, kernel_regularizer = l2(0.001)))
regressor.add(Dropout(0.1))

# Adding a third LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True, kernel_regularizer = l2(0.001)))
regressor.add(Dropout(0.1))

# # Adding a fourth LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units = 50, kernel_regularizer = l2(0.001)))
regressor.add(Dropout(0.1))

# Adding an output layer
regressor.add(Dense(units = 1))

# Compile the RNN
regressor.compile(optimizer = 'adam', loss = 'mae', metrics = ['mean_absolute_percentage_error'])

# Fitting the RNN to the Training set
regressor.fit(train_x, train_y, epochs = 50, shuffle = False, validation_data = (validate_x, validate_y))

# results = regressor.evaluate(validate_x, validate_y)
# metrics = regressor.metrics_names
# print('\nvalidation statistics: \n')
# for i in range(len(metrics)):
# 	print('{}: {}'.format(metrics[i],results[i]))

preds = regressor.predict(validate_x)
validate_x = np.reshape(validate_x, (validate_x.shape[0], validate_x.shape[2]))
test_dataset = np.concatenate((validate_x[:,0:5], preds), axis = 1)
test_dataset = sc.inverse_transform(test_dataset)
test_dataset = pd.DataFrame(test_dataset)
test_dataset.iloc[:,5] = round(test_dataset.iloc[:,5], 0)
print(test_dataset)

# validate_x = dummy_validate_x
# dummy_df = pd.DataFrame(validate_x)
# dummy_df['transformed_qty'] = sc.inverse_transform(preds)
# dummy_df.to_csv('dummy_qty.csv', index = True)
# print(dummy_df)

# transformed_pred = sc.inverse_transform(preds)

# print('\n-------------predictions accuracy-------------------')
# mape_results = mean_absolute_percentage_error(validate_y, preds)
# print('\nMAPE measure: ', mape_results)
# print('RMSE value: %0.3f' %np.sqrt(mean_squared_error(validate_y, preds)))

# validate_x = dummy_validate_x
# dates = validate_x.index.values

# plt.figure(figsize = (10,10))
# plt.plot(validate_y.resample('M'), color = 'red', label = 'actuals')
# plt.plot(preds, color = 'blue', label = 'predicted')
# plt.xticks(rotation = 90)
# plt.xlabel('dates')
# plt.ylabel('QTY')
# plt.legend(loc ='upper right')
# plt.show()
# # plt.show(block = False)
# # plt.pause(15)
