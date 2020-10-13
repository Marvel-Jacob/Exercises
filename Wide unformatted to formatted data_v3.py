import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
import time
import xlrd
import csv



start = time.time()


files = []
fieldnames = ["AE","Store Name","Store Category","Visit 1","Visit 2", "Visit 3", "Visit 4", "Visit 5", "Visit 6", "Total Nov",
"Visit 7","Visit 8", "Visit 9", "Visit 10", "Visit 11", "Visit 12", "Total Dec",
"Visit 13","Visit 14", "Visit 15", "Visit 16", "Visit 17", "Visit 18", "Total Jan", "Beat", "DB"]
for files_in_path in glob.glob(os.path.join(path_TS,'*.xlsx')):
    files.append(files_in_path)
# print(files)
# time.sleep(50)



# with open("C:/Users/Admin/Desktop/oct_master.csv","w", newline = "") as csv_file:
with open("C:/Users/Admin/Downloads/drive download/master.csv","w", newline = "", encoding = "utf-8") as csv_file:
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
				if len(data.columns)==23:
					needcols = data.columns[0:23]
					data = data[needcols]
					# print("\nThe columns are \n{}\n".format(data.columns))	
					data = data.fillna(0)
					# print(data.head(10))			

					for sn,sc,v1,v2,v3,v4,v5,v6,tn,v7,v8,v9,v10,v11,v12,td,v13,v14,v15,v16,v17,v18,tj in zip(list(data[data.columns[0]]),
						list(data[data.columns[1]]),list(data[data.columns[2]]),list(data[data.columns[3]]),list(data[data.columns[4]]),
						list(data[data.columns[5]]),list(data[data.columns[6]]),
						list(data[data.columns[7]]),list(data[data.columns[8]]),
						list(data[data.columns[9]]),list(data[data.columns[10]]),list(data[data.columns[11]]),
						list(data[data.columns[12]]),list(data[data.columns[13]]),list(data[data.columns[14]]),
						list(data[data.columns[15]]),list(data[data.columns[16]]),list(data[data.columns[17]]),
						list(data[data.columns[18]]),list(data[data.columns[19]]),list(data[data.columns[20]]),
						list(data[data.columns[21]]),list(data[data.columns[22]])):
						writer.writerow({"AE":ae_name,"Store Name":sn,"Store Category":sc,
							"Visit 1":v1,"Visit 2":v2, "Visit 3":v3, "Visit 4":v4, "Visit 5":v5, "Visit 6":v6, "Total Nov":tn,
							"Visit 7":v7, "Visit 8":v8, "Visit 9":v9, "Visit 10":v10, "Visit 11":v11, "Visit 12":v12, "Total Dec":td,
							"Visit 13":v13, "Visit 14":v14, "Visit 15":v15, "Visit 16":v16, "Visit 17":v17, "Visit 18":v18, "Total Jan":tj,
							 "Beat":beat_name, "DB":db_name})

				elif len(data.columns)==9:
					needcols = data.columns[0:9]
					data = data[needcols]
					# print("\nThe columns are \n{}\n".format(data.columns))
					data = data.fillna(0)
					for sn,sc,v1,v2,v3,v4,v5,v6,tj in zip(list(data[data.columns[0]]),list(data[data.columns[1]]),
						list(data[data.columns[2]]),list(data[data.columns[3]]),list(data[data.columns[4]]),
						list(data[data.columns[5]]),list(data[data.columns[6]]),
						list(data[data.columns[7]]),list(data[data.columns[8]])):
						writer.writerow({"AE":ae_name,"Store Name":sn,"Store Category":sc,
							"Visit 1":0,"Visit 2":0, "Visit 3":0, "Visit 4":0, "Visit 5":0, "Visit 6":0, "Total Nov":0,
							"Visit 7":0, "Visit 8":0, "Visit 9":0, "Visit 10":0, "Visit 11":0, "Visit 12":0, "Total Dec":0,
							"Visit 13":v1, "Visit 14":v2, "Visit 15":v3, "Visit 16":v4, "Visit 17":v5, "Visit 18":v6, "Total Jan":tj,
							 "Beat":beat_name, "DB":db_name})
				
				elif len(data.columns)==13:
					needcols = data.columns[0:9]
					data = data[needcols]
					# print("\nThe columns are \n{}\n".format(data.columns))
					data = data.fillna(0)
					for sn,sc,v1,v2,v3,v4,v5,v6,tj in zip(list(data[data.columns[0]]),list(data[data.columns[1]]),
						list(data[data.columns[2]]),list(data[data.columns[3]]),list(data[data.columns[4]]),
						list(data[data.columns[5]]),list(data[data.columns[6]]),
						list(data[data.columns[7]]),list(data[data.columns[8]])):
						writer.writerow({"AE":ae_name,"Store Name":sn,"Store Category":sc,
							"Visit 1":0,"Visit 2":0, "Visit 3":0, "Visit 4":0, "Visit 5":0, "Visit 6":0, "Total Nov":0,
							"Visit 7":0, "Visit 8":0, "Visit 9":0, "Visit 10":0, "Visit 11":0, "Visit 12":0, "Total Dec":0,
							"Visit 13":v1, "Visit 14":v2, "Visit 15":v3, "Visit 16":v4, "Visit 17":v5, "Visit 18":v6, "Total Jan":tj,
							 "Beat":beat_name, "DB":db_name})

				elif len(data.columns)==16:
					needcols = data.columns[0:16]
					data = data[needcols]
					# print("\nThe columns are \n{}\n".format(data.columns))
					data = data.fillna(0)
					for sn,sc,v1,v2,v3,v4,v5,v6,td,v7,v8,v9,v10,v11,v12,tj in zip(list(data[data.columns[0]]),list(data[data.columns[1]]),
						list(data[data.columns[2]]),list(data[data.columns[3]]),list(data[data.columns[4]]),
						list(data[data.columns[5]]),list(data[data.columns[6]]),
						list(data[data.columns[7]]),list(data[data.columns[8]]),
						list(data[data.columns[9]]),list(data[data.columns[10]]),list(data[data.columns[11]]),
						list(data[data.columns[12]]),list(data[data.columns[13]]),list(data[data.columns[14]]),
						list(data[data.columns[15]])):
						writer.writerow({"AE":ae_name,"Store Name":sn,"Store Category":sc,
							"Visit 1":0,"Visit 2":0, "Visit 3":0, "Visit 4":0, "Visit 5":0, "Visit 6":0, "Total Nov":0,
							"Visit 7":v1, "Visit 8":v2, "Visit 9":v3, "Visit 10":v4, "Visit 11":v5, "Visit 12":v6, "Total Dec":td,
							"Visit 13":v7, "Visit 14":v8, "Visit 15":v9, "Visit 16":v10, "Visit 17":v11, "Visit 18":v12, "Total Jan":tj,
							 "Beat":beat_name, "DB":db_name})


				else:
					break


end = time.time()

print("\nRun Time is {} minutes".format(round((end-start)/60),2))

