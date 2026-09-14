import os
import shutil

def fClearFolders(path):

    folder_path = path

    # Loop through all items in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # Check if it is a file and delete it
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def fClearSubFolders(path):

    folder_path = path

    # Loop through all items in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        try:
            # Check if it is a directory and delete it
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)

        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def fClearProjectFolders():
    # Clear Tables
    list = ('data','models','reports','reports/backtesting','reports/walkforward')
    for i in list:
        print(i)
        fClearFolders(i)

    backups = r'models/backup'
    fClearSubFolders(backups)
    print(f'subfolders in {backups} have been deleted')

def fMoveSamepleData():
    print('Moving Sample Data')