import shutil
import os
import glob

path="../Datasets/split-004F/test"

os.makedirs(path+'/OK', exist_ok=True)

for jpgfile in glob.iglob(os.path.join(path, "*_OK.jpg")):
    shutil.move(jpgfile, path+"/OK")

os.makedirs(path+'/NOK', exist_ok=True)
for jpgfile in glob.iglob(os.path.join(path, "*_FAIL.jpg")):
    shutil.move(jpgfile, path+"/NOK")
