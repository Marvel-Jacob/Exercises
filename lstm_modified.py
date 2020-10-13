import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from datetime import datetime

def parser(x):
	return datetime.strptime(x, '%Y-%m-%d')


data = pd.read_csv('/home/u73/workspace/dataset.csv')
data.set_index('Date', inplace = True)

#checking for null values
# print('\nchecking if any null values present in any feature....')
# time.sleep(1)
# print('columns name \t null values present')
# print(data.isnull().sum())

# values = data.values
# groups = [0, 1, 2, 3, 5, 6]
# i = 1
# # plot each column
# print('\ngraphically representing stationarity....')
# time.sleep(1)
# plt.figure()
# for group in groups:
# 	plt.subplot(len(groups), 1, i)
# 	plt.plot(values[:, group])
# 	plt.title(data.columns[group], y = 0.5, loc = 'right')
# 	i += 1
# time_graph = time.time()
# plt.show(block = False)
# plt.pause(5)
# plt.close()

no_of_rows = 1 if type(data) is list else data.shape[1]
df = pd.DataFrame(data)
cols, names = list(), list()
for i in range(1,0,-1):
	cols.append(df.shift(i))
	print(cols)
	names = names + [('var%d(t-%d)' % (j+1, i)) for j in range(no_of_rows)]
for i in range(0, 1):
		cols.append(df.shift(-i))
		print(cols)
		if i == 0:
			names = names + [('var%d(t)' % (j+1)) for j in range(no_of_rows)]
		else:
			names = names + [('var%d(t+%d)' % (j+1, i)) for j in range(no_of_rows)]
			print(names)


