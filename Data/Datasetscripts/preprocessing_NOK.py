import os
import re
import csv
import glob
import shutil

# assign directory
directory = "../Orginal"
 
# iterate over files in
# that directory
directory = directory+"/NOK_samples_xray"
targetpath="./Data/Preprocessed/NOK_samples_xray"
header=["part","failure"]
data=[]
for filename in os.listdir(directory):
    dat=[]
    dat.append(filename[4:])
    if(os.path.exists(directory+'/'+filename+'/Failure.txt')):
        with open(directory+'/'+filename+'/Failure.txt') as f:
            failure = f.readlines()
            dat.append(failure[0])
            f.close()
    else:
        dat.append("NaN")
        #with open(directory+'/'+filename+'/$Results_'+filename[4:]+'.log') as r:
        #    log = r.read()
        #    header= log[log.index("[HEADER]"):log.index("[CD]")]
        #    print(header)
        #    head= list(header[header.index("Job"):].split(","))
        #    for e in head:
        #           (a,b) = tuple(map(str,e.split(":")))
        #            head_tp[a]= b
        #    r.close()
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