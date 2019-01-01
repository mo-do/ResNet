import os
import re

def find_all_dirs_and_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)

def find_all_files(directory):
    dirs_and_files = find_all_dirs_and_files(directory)
    for file in dirs_and_files:
        if os.path.isfile(file):
            yield file

def find_all_dirs(directory):
    dirs_and_files = find_all_dirs_and_files(directory)
    for dir in dirs_and_files:
        if os.path.isdir(dir):
            yield dir

def filePath2fileName(filePath, include_extended=True):
    fileName = os.path.basename(filePath)
    if not include_extended:
        fileName = fileName.rsplit('.', 1)[0]
    return fileName