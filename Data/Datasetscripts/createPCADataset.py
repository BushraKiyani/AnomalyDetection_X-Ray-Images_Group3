import os
import glob
import shutil

os.makedirs('../Datasets/PCA/PCA_com', exist_ok=True)

os.makedirs('../Datasets/PCA/PCA_com/NOK', exist_ok=True)
for filename in os.listdir("../Preprocessed/NOK_samples_xray"):
    for jpgfile in glob.iglob(os.path.join('../Preprocessed/NOK_samples_xray/'+filename, "*_FAIL.jpg")):
        shutil.copy(jpgfile,'../Datasets/PCA/PCA_com/NOK')

os.makedirs('../Datasets/PCA/PCA_com/OK', exist_ok=True)
for filename in os.listdir("../Preprocessed/OK_samples_xray"):
    for jpgfile in glob.iglob(os.path.join('../Preprocessed/OK_samples_xray/'+filename, "*_OK.jpg")):
        shutil.copy(jpgfile,'../Datasets/PCA/PCA_com/OK')