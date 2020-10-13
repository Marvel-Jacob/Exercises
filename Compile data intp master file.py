import pandas as pd
from datetime import datetime
import glob
import os
import sys
import time
import csv
import pyautogui
import shutil

select_month = datetime(2020, 7, 1, 0, 0)

def create_master_file():
    start = time.time()

    files = []

    path = 'C:/Users/Admin/Downloads/drive download/November/'
    for files_in_path in glob.glob(os.path.join(path, '*.xlsx')):
        files.append(files_in_path)

    manager_m = {}
    cpt = pd.read_csv('C:/Users/Admin/Desktop/reps/MAIN dummy rep/DAILY/dataset.csv')
    for ae, state, am, sam, rm in zip(cpt['AE'], cpt['State'], cpt['AM'], cpt['SAM'], cpt['RM']):
        manager_m.update({ae: {"state": state, "am": am, "sam": sam, "rm": rm}})

    with open("C:/Users/Admin/Desktop/reps/MAIN dummy rep/DAILY/master.csv", "w", newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for file in files:
            ae_name = file.split('-')[1].strip().split('.xl')[0]
            if "_" in ae_name:
                ae_name = ae_name.strip('_')
            print("Accessing {}'s file".format(ae_name))
            book = pd.ExcelFile(file)
            sheet = [sheet for sheet in book.sheet_names if "month" in sheet.lower() and "copy" not in sheet.lower()][0]
            temp_data = pd.read_excel(file, sheet_name=sheet, skiprows=0)
            si = int(list(temp_data.columns).index(select_month))
            ei = si + 6
            data = pd.read_excel(file, sheet_name=sheet, skiprows=1)
            data = data.fillna("")
            needcols = data.columns[si:ei]
            # continue
            # if len(data.columns)==18:
            # 	needcols = data.columns[12:18]
            # elif len(data.columns)==30:
            # 	needcols = data.columns[24:30]
            # elif len(data.columns)==6:
            # 	needcols = data.columns[0:6]
            # elif len(data.columns)==24:
            # 	needcols = data.columns[18:24]
            # print("Selected columns {} \n".format(needcols))
            data = data[needcols]  # shift the columns to get the NOV data
            # print(data.head(5))
            # time.sleep(1)
            date = data.columns[0]
            beat_name = data.columns[1]
            db_name = data.columns[2]
            tc_value = data.columns[3]
            tp_value = data.columns[4]
            tb_value = data.columns[5]
            for dt, bn, dbn, tc, tp, tb in zip(list(data[date]),
                                               list(data[beat_name]),
                                               list(data[db_name]),
                                               list(data[tc_value]),
                                               list(data[tp_value]),
                                               list(data[tb_value])):
                if ae_name in manager_m:
                    state = manager_m[ae_name]["state"]
                    rm = manager_m[ae_name]["rm"]
                    sam = manager_m[ae_name]["sam"]
                    am = manager_m[ae_name]["am"]
                    writer.writerow({"State": state, "RM": rm, "SAM": sam, "AM": am, "AE": ae_name,
                                     "Date": dt, "Beat Name": bn, "DB Name": dbn, "TC": tc, "TP": tp, "TB": tb})
                # try:
                # 	writer.writerow({"State":state,"RM":rm,"SAM":sam,"AM":am,"AE":ae_name,
                # 		"Date":dt,"Beat Name":bn,"DB Name":dbn,"TC":int(tc),"TP":int(tp),"TB":int(tb)})
                # except:
                # 	writer.writerow({"State":state,"RM":rm,"SAM":sam,"AM":am,"AE":ae_name,
                # 		"Date":dt,"Beat Name":bn,"DB Name":dbn,"TC":0,"TP":0,"TB":0})
                # writer.writerow({"AE":ae_name,"Date":dt,"Beat Name":bn,"DB Name":dbn,"TC":tc,"TP":tp,"TB":tb})
                else:
                    writer.writerow(
                        {"State": "ADD INFO/DATA FOR THIS AE", "RM": "ADD RM DATA", "SAM": "ADD SAM DATA",
                         "AM": "ADD AM DATA", "AE": ae_name})
                    break

    print("Once step running...")
    data = pd.read_csv("C:/Users/Admin/Desktop/reps/MAIN dummy rep/DAILY/master.csv")
    data['Running Total'] = 0
    # cumsum_data = data[['AE','Date','TB']].groupby(['AE','Date']).sum().groupby("AE").cumsum().reset_index()
    # cumsum_data = cumsum_data.rename(columns = {"TB":"Running Total"})
    # temp_rt = []
    # for d, rt in list(zip(cumsum_data['Date'],cumsum_data["Running Total"])):
    # 	if int(d)==int(D) or int(d)>int(D):
    # 		temp_rt.append(0)
    # 	else:
    # 		temp_rt.append(rt)
    # cumsum_data['Running Total'] = temp_rt
    # data = pd.merge(data,cumsum_data, on = ['AE','Date'])
    # # print(cumsum_data.head(45))
    data.to_csv('C:/Users/Admin/Desktop/reps/MAIN dummy rep/DAILY/master.csv', index=False)
    end = time.time()
    print("\n{} minutes to make master file".format(round((end - start) / 60, 2)))

def open_history_tabs_in_chrome():
    chrome_menu = [1345, 51]
    history = [1245, 163]
    open_tabs = [916, 214]
    new_tab = [1162, 17]
    go_to_link = [224, 81]
    download_folder = [342, 349]
    time.sleep(1)
    pyautogui.click(x=chrome_menu[0], y=chrome_menu[1])
    time.sleep(1)
    pyautogui.click(x=history[0], y=history[1])
    time.sleep(1)
    pyautogui.click(x=open_tabs[0], y=open_tabs[1])
    time.sleep(1)
    pyautogui.click(x=new_tab[0], y=new_tab[1])
    time.sleep(1)
    pyautogui.click(x=go_to_link[0], y=go_to_link[1])
    time.sleep(8)
    pyautogui.click(x=download_folder[0], y=download_folder[1], button="right")
    while True:
        points = pyautogui.locateCenterOnScreen('download_option_chrome.png')
        if points != None:
            x, y = points
            break
    pyautogui.click(x=x, y=y)


def open_chrome():
    chrome = [558, 751]
    pyautogui.click(x=chrome[0], y=chrome[1])
    time.sleep(5)

def direct_download_operation():
    go_to_link = [224, 81]
    new_tab = [1162, 17]
    download_folder = [342, 349]
    download_button = [456, 638]
    pyautogui.click(x=new_tab[0], y=new_tab[1])
    time.sleep(2)
    pyautogui.click(x=go_to_link[0], y=go_to_link[1])
    time.sleep(6)
    pyautogui.click(x=download_folder[0], y=download_folder[1], button="right")
    while True:
        points = pyautogui.locateCenterOnScreen('download_option_chrome.png')
        if points != None:
            x, y = points
            break
    pyautogui.click(x=x, y=y)

def open_folder_containing_downloaded_file():
    navigate = [206,700]
    pyautogui.click(x=navigate[0], y=navigate[1])
    navigate = [232,630]
    time.sleep(1)
    pyautogui.click(x=navigate[0], y=navigate[1])
    time.sleep(1)

def delete_existing_files():
    path = "C:\\Users\\Admin\\Downloads\\drive download\\"
    files = glob.glob(os.path.join(path, 'November*'))
    if len(files)==2:
        shutil.rmtree(files[0])
        os.remove(files[1])

def move_file_and_extract():
    while True:
        points = pyautogui.locateCenterOnScreen('download_file.png')
        if points!=None:
            x,y = points
            break
    pyautogui.click(x=x, y=y, button="right")
    time.sleep(1)
    x,y = pyautogui.locateCenterOnScreen('cut.png')
    pyautogui.click(x=x, y=y)
    time.sleep(1)
    while True:
        points = pyautogui.locateCenterOnScreen('drive folder.png')
        if points!=None:
            x,y = points
            pyautogui.doubleClick(x=x, y=y)
            break
    time.sleep(1)
    x, y = pyautogui.locateCenterOnScreen('paste.png')
    pyautogui.click(x=x, y=y)
    time.sleep(5)
    while True:
        points = pyautogui.locateCenterOnScreen('download_file.png')
        if points!=None:
            x,y = points
            break
    pyautogui.click(x= x, y= y, button="right")
    time.sleep(1)
    while True:
        points = pyautogui.locateCenterOnScreen('extract_here.png')
        if points!=None:
            x, y = pyautogui.locateCenterOnScreen('extract_here.png')
            break
    pyautogui.click(x= x, y= y)
    time.sleep(3)

def back_to_terminal():
    time.sleep(1)
    terminal_location = [721, 745]
    pyautogui.click(x=terminal_location[0], y=terminal_location[1])


if __name__=="__main__":
    if len(sys.argv)>1:
        if sys.argv[1]=="-automate":
            start = time.time()
            delete_existing_files()
            open_chrome()
            try:
                if sys.argv[2]=="-history":
                    open_history_tabs_in_chrome()
            except:
                direct_download_operation()
            pyautogui.alert('Wait for the download to complete!! Then click on "OK"', title="WAIT!")
            time.sleep(2)
            open_folder_containing_downloaded_file()
            move_file_and_extract()
            back_to_terminal()
            create_master_file()
            end = time.time()
            print('\nProgram Exectution Time: {} minutes'.format(round((end-start)/60, 2)))
    else:
        create_master_file()


