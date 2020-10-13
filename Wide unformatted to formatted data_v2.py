import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
import time
import xlrd
import csv

start = time.time()
# path = 'C:/Users/Admin/Downloads/drive download/October/'
path = 'C:/Users/Admin/Downloads/drive download/November/'
files = []
fieldnames = ["AE","Store Name","Store Category","Visit 1","Visit 2", "Visit 3", "Visit 4", "Visit 5", "Visit 6", "Beat", "DB"]
for files_in_path in glob.glob(os.path.join(path,'*.xlsx')):
    files.append(files_in_path)
# print(files)
# time.sleep(50)



# with open("C:/Users/Admin/Desktop/oct_master.csv","w", newline = "") as csv_file:
with open("C:/Users/Admin/Desktop/Exercises/master.csv","w", newline = "", encoding = "utf-8") as csv_file:
	writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
	writer.writeheader()
	for file in files:
		# ae_name = file.split('-')[2].strip().split('.xl')[0]
		ae_name = file.split('-')[1].strip().split('.xl')[0]
		if "_" in ae_name:
			ae_name = ae_name.strip('_')
		print("\n------------------------------------------------------------------------------------------------------------------\n")
		print(ae_name)
		# data = pd.read_excel(file,skiprows=1)
		# print("\n",ae_name + " - "+str(list(data.columns))+ " - "+str(len(list(data.columns))))
		book = xlrd.open_workbook(file)
		sheets = [sheet for sheet in book.sheet_names()]
		for sheet in sheets:
			if "month" not in sheet.lower() and "ample" not in sheet.lower() and "sheet" not in sheet.lower():
				# print(sheet)
				data = pd.read_excel(file, sheet_name = sheet)
				data = data.fillna(" ")
				data.columns = [str(i) for i in data.columns]
				# print(data.head(10))
				print("\nSheet: "+sheet+ " , "+"No: of columns are: "+str(len(data.columns)))
				print('---------------------------------------------------')
				if len(data.columns)>1:
					# print(data.head(10))
					if "beat" in str(data.iloc[2][0]).lower():
						beat_name = str(data.iloc[2][1]).upper()
					else:
						beat_name = "x"
					if "db" in str(data.iloc[3][0]).lower():
						db_name = str(data.iloc[3][1]).upper()
					else:
						db_name = "x"
					# print(beat_name +" - "+ db_name) 
					data = pd.read_excel(file, sheet_name = sheet, skiprows = 8, nrows = 35)
					data.columns = [str(i) for i in data.columns]
					icols = {i:n for i,n in zip(list(data.columns),[i for i in range(len(data.columns))])}
					print("\nThe columns are \n{}\n".format(icols))
					n = str(input("Enter start of range or enter 'p' to skip: "))
					if (n!="p" and n!="P"):
						range_list = [i for i in range(int(n),int(n)+6)]
						# print(range_list)
					else:
						range_list = ["p","P"]
					colmap = [0,1]
					colmap.extend(range_list)
					# print(colmap)
					if ("p" or "P") in colmap:
						pass
					else:
						needcols = [x for x,y in icols.items() for z in colmap if str(z)==str(y)]
						# print(needcols)
						data = data[needcols]	
						data = data.fillna(0)
						# print(data.columns)			

						for sn,sc,v1,v2,v3,v4,v5,v6 in zip(list(data[data.columns[0]]),list(data[data.columns[1]]),
							list(data[data.columns[2]]),list(data[data.columns[3]]),list(data[data.columns[4]]),
							list(data[data.columns[5]]),list(data[data.columns[6]]),
							list(data[data.columns[7]])):
							writer.writerow({"AE":ae_name,"Store Name":sn,"Store Category":sc,
								"Visit 1":v1,"Visit 2":v2, "Visit 3":v3, "Visit 4":v4, "Visit 5":v5, "Visit 6":v6,
								 "Beat":beat_name, "DB":db_name})
			else:
					pass

end = time.time()

print("\nRun Time is {} minutes".format(round((end-start)/60),2))

