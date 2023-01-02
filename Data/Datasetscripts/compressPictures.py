from PIL import Image
import PIL
import shutil
import os
import glob

path="../Datasets/split-004F/"
folder = "test/NOK"
os.makedirs(path +"/skip/"+ folder, exist_ok=True)

for picture in os.listdir(path+folder):
    img = Image.open(path +folder+ "/" +picture)
    #print("Original size:" +str(img.size))
    img = img.crop((1504, 654, 2016, 1166))
    img = img.resize((512,512))
    #print("Rezised:" +str(img.size))
    img.save(path +"/skip/"+ folder+ "/" + picture)
