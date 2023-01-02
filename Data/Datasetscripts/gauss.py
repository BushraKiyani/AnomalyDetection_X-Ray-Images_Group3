import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#for picture in os.listdir("./test"):
picture = mpimg.imread('../Preprocessed/split/test/3137.232.527_2022.02.03_8017_004_new set.3_FAIL.jpg')

kernel= 50
alpha = 10

filter_blurred_f = ndimage.gaussian_filter(picture, kernel)
sharpened = picture + alpha * (picture - filter_blurred_f)

#blurred_f = ndimage.gaussian_filter(picture, 5)
#filter_blurred_f = ndimage.gaussian_filter(blurred_f, kernel)
#sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)

plt.figure(figsize=(12, 4))
plt.subplot(121)
#plt.subplot(131)
plt.imshow(picture, cmap=plt.cm.gray)
plt.axis('off')
plt.title(label="original", loc='center')
#plt.subplot(132)
#plt.imshow(blurred_f, cmap=plt.cm.gray)
#plt.axis('off')
#plt.title(label="blurred", loc='center')
plt.subplot(122)
#plt.subplot(133)
plt.imshow(sharpened, cmap=plt.cm.gray)
plt.axis('off')
label = "kernel=" + str(kernel) +", alpha="+ str(alpha)
plt.title(label=label, loc='center')

plt.tight_layout()
plt.show()

