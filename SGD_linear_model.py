import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess 

def outlier_remove(data):
	min_quan= data['Actual'].quantile(0.25)
	max_quan= data['Actual'].quantile(0.75)
	IQR = max_quan - min_quan
	# print(min_quan, max_quan, IQR)
	max_quan = max_quan + (1.5 * IQR)
	# print(min_quan, max_quan)
	data = data[data['Actual'] > min_quan]
	data = data[data['Actual'] < max_quan]
	return data  

# subprocess.call('/usr/bin/python3 ~/workspace/tableau/python_scripts/mongo_to_csv.py', shell = True)

 
dataset = pd.read_csv('/home/u73/workspace/tableau/python_scripts/new_data_transformed_metrics.csv', low_memory = False)
# dataset = outlier_remove(dataset)
# dataset.to_csv('eg.csv', index = False)


# print('range of actuals {} to {}'.format(min(dataset['Actual']),max(dataset['Actual'])))

dataset['Date'] = dataset['Date'].apply(lambda x: str(x).split(' ')[0])
dataset = dataset[['Date','Session','Item','Actual','Predicted']]
dataset = dataset.query('Item == "ANDHRA VEG MEALS" and Session == "Morning" and Date >= "2019-04-01"')
dataset = dataset.groupby(['Date']).agg({'Actual':'sum', 'Predicted':'sum'})
# dataset.to_csv('eg.csv', index = False)

plt.plot(dataset['Actual'], color = 'red', label = 'actuals')
plt.plot(dataset['Predicted'], color = 'blue', label = 'predicted')
dates = dataset.iloc[:,1]
plt.xticks(rotation = 90)
plt.xlabel('dates')
plt.ylabel('QTY')
plt.legend(loc ='upper right')
plt.show()
