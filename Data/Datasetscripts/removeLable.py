
import os


path="../Datasets/split-004F/test/OK"

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

for picture in os.listdir(path):
    nth = find_nth(picture,"_",4)
    part = picture[:nth]
    newname = part+".jpg"
    if(newname in os.listdir(path) ):
        os.remove(picture)
    else:
        os.rename(path+'/'+picture, path+'/'+newname)