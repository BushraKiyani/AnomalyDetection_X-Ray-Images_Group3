import os
import pandas as pd

# assign directory
nok = "../Preprocessed/NOK_samples_xray/004"

ok_csv = pd.read_csv("../Preprocessed/OK_samples_xray/data.csv")
nok_csv= pd.read_csv("../Preprocessed/NOK_samples_xray/data.csv")

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

for picture in os.listdir(nok):
    nth = find_nth(picture,"_",3)
    part = picture[:nth]
    index1 = picture.rindex("_")
    index2 = picture.rindex(".")
    status = picture[index1+1:index2]
    if(status == "OK"):
        nok_csv.loc[nok_csv["part"] == part, "failure"] = "NONE"

rel_nok = nok_csv.loc[nok_csv["failure"] != "NONE"]

#nok
train_nok = rel_nok.sample(frac = 2/3,random_state=200)
train_nok = train_nok.append(nok_csv.drop(rel_nok.index).sample(frac = 2/3))
print("train_nok", len(train_nok))
test_nok = nok_csv.drop(train_nok.index)
print("test_nok", len(test_nok))

#ok
train_ok = ok_csv.sample(frac = 2/3,random_state=200)
print("train_ok", len(train_ok))
test_ok = ok_csv.drop(train_ok.index)
print("test_ok", len(test_ok))

#combined
train = train_nok.append(train_ok, ignore_index=True)
print("train", len(train))
test = test_nok.append(test_ok, ignore_index=True)
print("test", len(test))

#write csv
os.makedirs('../Datasets/split-004F', exist_ok=True)
train.to_csv('../Datasets/split-004F/train.csv',index=False)
test.to_csv('../Datasets/split-004F/test.csv',index=False)
