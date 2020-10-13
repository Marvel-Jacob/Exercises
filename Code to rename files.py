import os
import time
import sys

start = time.time()
files = []
default_path = "C:/Users/Admin/Downloads/drive download/November/"

def renamer(path):
	try:
		ext = sys.argv[1]
		rename_ext = ext+"_"
		try:
			for file in os.listdir(path_of_files):
				os.rename(path_of_files+file,path_of_files+rename_ext+file)
			print("\nDONE!!")
		except:
			print("\nFile is open. Please close it to continue (run code again)")

		end = time.time()
		print("\nRun Time is {} minutes".format(round((end-start)/60),2))

	except:
		print("\nAdd an argument, which will concatenate with the files")

path_of_files = input(str("Enter the path of the files: "))
if ':' and '/' in path_of_files:
	renamer(path_of_files)
elif "\\" in path_of_files:
	path_of_files = path_of_files.replace("\\","/",20)+'/'
	print("Correct Path is: ",path_of_files)
	renamer(path_of_files)
elif '/' not in path_of_files:
	print("Not a defined path...try again!")






