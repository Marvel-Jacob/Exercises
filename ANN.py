import pandas as pd
import numpy as np
import keras
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop,Adam
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import mean_squared_error


dataset = pd.read_csv('/home/u73/workspace/dataset.csv')
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset.set_index('Date', inplace = True)
# print(data.columns)
# data.set_index('ItemCode', inplace = True)
x = dataset.iloc[:,0:5]
y = dataset.iloc[:,5]
# print(x.columns)
# print('-----')
# print(y)

train_x, validate_x, train_y, validate_y = train_test_split(x,y,test_size = 0.1, shuffle = False)
# print(train_x.shape)
# print(train_y.shape)
# print(validate_x.shape)
# print(validate_y.shape)

# print(train_x.head())
# print(train_y.head())
# print(validate_x.head())
# print(validate_y.head())

model = keras.Sequential()
model.add(Dense(100, activation = tf.nn.relu, input_dim = len(x.columns)))
model.add(Dropout(0.2))
model.add(Dense(64, activation = tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(50, activation = tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(30, activation = tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'linear'))

optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(optimizer = 'adam', loss='mean_squared_error',metrics=['mean_absolute_error', 'mean_squared_error'])
model.fit(train_x, train_y, validation_data = (validate_x, validate_y), epochs = 100)

preds = model.predict(validate_x)

# print(validate_x ,' - ',preds)
# validate_x['qty'] = validate_y
validate_x['pred_qty'] = preds
print(validate_x)
# validate_x['rmse'] = np.sqrt(mean_squared_error(validate_y,predictions))
# validate_x.to_csv('Test_predictions.csv', index = False)
print('RMSE : ',np.sqrt(mean_squared_error(validate_y,preds)))
# print(np.sqrt(mean_squared_error(validate_y,predictions)))
# print('length of pred', len(predictions))
# pred = pd.DataFrame()
# pred
