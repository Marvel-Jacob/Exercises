import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import sys
import csv


path_2_data = sys.argv[1]
if path_2_data.split('.')[1] == 'csv':
	data = pd.read_csv(path_2_data)
elif path_2_data.split('.')[1] == 'xlsx':
	data = pd.read_excel(path_2_data)
print('\nColumns of the data')
print(data.columns)
col1_index = int(input('Enter the 1st column index '))
col1 = data.columns.to_list()[col1_index]
col2_index = int(input('Enter the 2nd column index '))
col2 = data.columns.to_list()[col2_index]
col1_values = [i for i in data[col1].to_list() if str(i) != 'nan']
col2_values = [i for i in data[col2].to_list() if str(i) != 'nan']
# col1_values = data[col1].to_list()
# col2_values = data[col2].to_list()
print('\nLength of Column1: ',len(col1_values),' and',' Length of Column2: ',len(col2_values))
if len(col1_values) < len(col2_values):
	c = 0
	val1 = []
	val2 = []
	val3 = []
	for i in col1_values:
		c = c + 1
		print(c,' iteration')
		for j in col2_values:
			ratio = SequenceMatcher(None,str(i).lower().strip(),str(j).lower().strip()).ratio()
			if ratio > 0.80:
				val1.append(i)
				val2.append(j)
				val3.append(round(ratio,2))
	data[col1+'_mod'] = pd.Series(val1)
	data[col2+'_mod'] = pd.Series(val2)
	data['ratio'] = pd.Series(val3)
	data.to_csv('/home/u73/Desktop/'+sys.argv[1].split('/')[-1], index = False)
else:
	c = 0
	val1 = []
	val2 = []
	val3 = []
	for i in col2_values:
		c = c + 1
		print(c, ' iteration')
		for j in col1_values:
			ratio = SequenceMatcher(None,str(i).lower().strip(),str(j).lower().strip()).ratio()
			if ratio > 0.80:
				val1.append(i)
				val2.append(j)
				val3.append(round(ratio,2))
	data[col1+'_mod'] = pd.Series(val1)
	data[col2+'_mod'] = pd.Series(val2)
	data['ratio'] = pd.Series(val3)
	data.to_csv('/home/u73/Desktop/'+sys.argv[1].split('/')[-1], index = False)

# print(SequenceMatcher(None,'Apple','pple').ratio())