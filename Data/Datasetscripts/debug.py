import os
import re
import csv
import glob
import shutil

targetpath = "../Preprocessed/OK_samples_xray"
filename= "3137.232.529_2022.01.27_3342"
for jpgfile in glob.iglob("../Orginal/OK_samples_xray/"+ filename+ "*_004_*.jpg"):
    shutil.copy(jpgfile, targetpath+"/004")