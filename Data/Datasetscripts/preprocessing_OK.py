import os
import re
import csv
import glob
import shutil

# assign directory
directory = "../Orginal"
 
# iterate over files in
# that directory
directory = directory+"/OK_samples_xray"
targetpath="./Data/Preprocessed/OK_samples_xray"
header=["part","failure"]
data=[]
for filename in os.listdir(directory):
    dat=[filename, "NONE"]
    data.append(dat)
    for jpgfile in glob.iglob(os.path.join(directory+'/'+filename, "*_001_*.jpg")):
        shutil.copy(jpgfile, targetpath+"/001")
    for jpgfile in glob.iglob(os.path.join(directory+'/'+filename, "*_002_*.jpg")):
        shutil.copy(jpgfile, targetpath+"/002")
    for jpgfile in glob.iglob(os.path.join(directory+'/'+filename, "*_003_*.jpg")):
        shutil.copy(jpgfile, targetpath+"/003")
    for jpgfile in glob.iglob(os.path.join(directory+'/'+filename, "*_004_*.jpg")):
        shutil.copy(jpgfile, targetpath+"/004")
with open(targetpath+"/data.csv", 'w') as csvf:
    writer = csv.writer(csvf)
    writer.writerow(header)
    writer.writerows(data)
    csvf.close()