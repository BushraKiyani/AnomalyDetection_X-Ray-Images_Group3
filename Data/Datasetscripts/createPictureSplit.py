import os
import pandas as pd
import glob
import shutil

path="../Datasets/split"
file="001"

def copyimage(folder,csv,target):
    print(folder,target)
    count=0
    for picture in csv["part"]:
        for jpgfile in glob.iglob(str(folder) + "/" + str(picture) + "_*.jpg"):
            shutil.copy(jpgfile, target)
            count +=1
    print(count)

#train
os.makedirs(path+'/train', exist_ok=True)
train = pd.read_csv("../Datasets/train.csv")

copyimage("../Preprocessed/NOK_samples_xray/"+file, train, path+'/train')
copyimage("../Preprocessed/OK_samples_xray/"+file, train, path+'/train')

#test
os.makedirs(path+'/test', exist_ok=True)
test = pd.read_csv("../Datasets/test.csv")

copyimage("../Preprocessed/NOK_samples_xray/"+file, test, path+'/test')
copyimage("../Preprocessed/OK_samples_xray/"+file, test, path+'/test')