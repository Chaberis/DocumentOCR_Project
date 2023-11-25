import os


directory = "dataset/handwritten"
files = os.listdir(directory)
files.sort()

for i, filename in enumerate(files):
    newfilename = f'handwritten{i}{os.path.splitext(filename)[1]}'
    oldpath = os.path.join(directory, filename)
    newpath = os.path.join(directory, newfilename)
    os.rename(oldpath, newpath)
