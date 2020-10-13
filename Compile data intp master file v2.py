import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
import time
import xlrd
import csv
import multiprocessing


needcols = []

start = time.time()
path = 'C:/Users/Admin/Downloads/drive download/November/'
files = []
appender = {}
fieldnames = ["AE","Beat Name", "DB Name","Store Name","Store Category","Date","Qty"]
for files_in_path in glob.glob(os.path.join(path,'*.xlsx')):
    files.append(files_in_path)


with open("C:/Users/Admin/Downloads/drive download/master.csv","w", newline = '') as csv_file:
	writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
	writer.writeheader()
	for file in files[69:]:
		ae_name = file.split('-')[1].strip().split('.xl')[0]
		if "_" in ae_name:
			ae_name = ae_name.strip('_')
		print("\n")
		print(ae_name)
		# print("\n",ae_name + " - "+str(list(data.columns))+ " - "+str(len(list(data.columns))))
		book = xlrd.open_workbook(file)
		sheets = [sheet for sheet in book.sheet_names()]
		for sheet in sheets:
			if "month" not in sheet.lower() and "ample" not in sheet.lower() and "sheet" not in sheet.lower():
				print(sheet)
				data = pd.read_excel(file, sheet_name = sheet, header = None)
				beat_name = str(data.iloc[3][1]).upper()
				db_name = str(data.iloc[4][1]).upper()
				data = pd.read_excel(file, sheet_name = sheet, skiprows = 3)
				# print(data.columns)
				data = data.fillna(" ")
				data.columns = [str(i) for i in data.columns]
				data = pd.read_excel(file, sheet_name = sheet, skiprows = 8, nrows = 50)
				# data = pd.melt(data, id_vars = ["Store Name","Store Category"], var_name = "Date", value_name = "Quantity")
				for sn,sc in zip(list(data["Store Name"]),
					list(data["Store Category"])):
					# writer.writerow({"AE":ae_name,"Beat Name":beat_name, "DB Name":db_name,"Store Name":sn,"Store Category":sc,
					# 	"Date":dt,"Qty":qty})
					writer.writerow({"AE": ae_name, "Beat Name": beat_name, "DB Name": db_name, "Store Name": sn,
									 "Store Category": sc})

				# appender.update({"AE":ae_name, "Beat Name":beat_name, "DB Name":db_name,
				# 	})

end = time.time()

print("\nTime taken in minutes {}".format(round((end-start)/60),3))


# data = pd.read_excel('C:/Users/Admin/Downloads/drive download/master.xlsx')
# data['Date'] = pd.to_datetime(data['Date'])
# data.excel('C:/Users/Admin/Downloads/drive download/master_mod.xlsx', inplace = True)