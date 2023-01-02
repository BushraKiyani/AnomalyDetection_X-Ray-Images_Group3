import numpy as np
import pandas as pd
import cv2 as cv
import cv2 as cv2
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage import measure
from skimage import data, filters, color, morphology
from skimage.morphology import disk
import matplotlib
from os import listdir
from os.path import isfile, join


matplotlib.rcParams['figure.figsize'] = [10, 10]

matplotlib.rcParams['figure.figsize'] = [10, 10]

#"""In the following i loaded data from one day so, that there is as least difference between train and test as possible."""

# configure your paths
#onlyfiles = [f for f in listdir('../Data/Datasets/split-004F/train/OK/') if isfile(join('../Data/Datasets/split-004F/train/OK/', f))]
onlyfiles = listdir('../Data/Datasets/01.28/OK/')
#onlyfiles_test = [f for f in listdir('../Data/Datasets/split-004F/train/NOK/') if isfile(join('../Data/Datasets/split-004F/train/NOK/', f))]
onlyfiles_test = listdir('../Data/Datasets/01.28/NOK/')
my_path = '../Data/Datasets/01.28/OK/'
my_path_test = '../Data/Datasets/01.28/NOK/'


Title_Info_Good = pd.DataFrame([tmp_str.split('_') for tmp_str in onlyfiles]).rename(columns = {0: 'Code_A',  
                                                                                   1: 'Date',
                                                                                   2: 'Code_B',
                                                                                   3: 'Angle',
                                                                                   4: 'Set',
                                                                                   5: 'Res'})
Title_Info_Good['Date'] = pd.to_datetime(Title_Info_Good.Date)
Title_Info_Good['Res'] = Title_Info_Good.Res.str.split('.', expand = True)[0]
Title_Info_Good['Set'] = Title_Info_Good.Set.str.split('.', expand = True)[1]

Title_Info_Anomal = pd.DataFrame([tmp_str.split('_') for tmp_str in onlyfiles_test]).rename(columns = {0: 'Code_A',  
                                                                                   1: 'Date',
                                                                                   2: 'Code_B',
                                                                                   3: 'Angle',
                                                                                   4: 'Set',
                                                                                   5: 'Res'})
Title_Info_Anomal['Date'] = pd.to_datetime(Title_Info_Anomal.Date)
Title_Info_Anomal['Res'] = Title_Info_Anomal.Res.str.split('.', expand = True)[0]
Title_Info_Anomal['Set'] = Title_Info_Anomal.Set.str.split('.', expand = True)[1]

ind_bool_train = (Title_Info_Good.Date == dt.datetime(2022, 1, 28)).values
files_rel_train = [onlyfiles[i] for i in range(len(onlyfiles)) if ind_bool_train[i] ]
image_list_train = [cv2.imread( my_path + img_name, cv2.IMREAD_GRAYSCALE) for  img_name in files_rel_train[0:50]]
ind_bool_test = (Title_Info_Anomal.Date == dt.datetime(2022, 1, 28))
files_rel_test = [onlyfiles_test[i] for i in range(len(onlyfiles_test)) if ind_bool_test[i] ]
image_list_test = [cv2.imread( my_path_test + img_name, cv2.IMREAD_GRAYSCALE) for  img_name in files_rel_test]

"""In this loaded image_list_test, there are like 3 anomalies which are relative nice to find. These are in the images with the indeces 3, 5, 6 and are all at the right edge in the upper half. Around y = 1000. The image names say there should be errors in the images 1 and 8.... and the rest is labled ok..
Of these test images there is also to recognize, that the images 3 and 4 are a lot harder to reconstruct because there angle differ from the train files. 
"""

"""## General Preprocessing

In this section preprocessing methods are listed and / or defined which are applied on all data, the train and the test data. This mostly refer to improve the image quality. 

The following procedures i looked at:
- Histogramm equlization
- Truncate the values and then rescale

At the first glance i didnt saw a big improvement but maybe just doit again when all stand.

## Preprocess train images

Here methods are mentioned to improve the train procedure. This mostly refer to the decision of which images are used for train. All images for one angle can be used theoretically, but that will cause memory issues and also not improve the method so much. The important point is to use the most diverse images for train.
"""

def getDiverseImages(image_name_list, path, max_mean_diff):
  img_list_great_diff = []
  img_names_great_diff = []
  for img_name in image_name_list:
    tmp_image = cv2.imread( path + img_name, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if len(img_list_great_diff) == 0:
      img_list_great_diff.append(tmp_image)
      img_names_great_diff.append(img_name)

    else:
      new_img = True
      for i in range(len(img_list_great_diff)):
        MSE = ((img_list_great_diff[i].astype('float32') - tmp_image.astype('float32'))**2).mean()
        if MSE < max_mean_diff:
          new_img = False
          break
      if new_img:
        img_list_great_diff.append(tmp_image)    
        img_names_great_diff.append(img_name)
  return(img_names_great_diff)

"""## Preprocess test data

Here functions are defined which are used only for the test procedure. Goal is to transform the test data so, that the reconstruction algorithm can get the best out of it. 

- Rotation 
- Shearing
- Similarity measure
"""

# findBestOrientation(image, img_List_Comp,  x_span, y_span, x_shear_span, y_shear_span, step_per_dir)
# image: image matrix 252x252
# img_List_Comp: list of images 252x252
# x_span, y_span: list of 2 integers, the min and max shift one direction negative the other positive
# x_shear_span, y_shear_span: list of 2 floats, the min and max shear in x and y direction


def findBestOrientation(image, img_List_Comp,  x_span, y_span, x_shear_span, y_shear_span, step_per_dir):

  # Define the parametergrid

  x_shear = [x_shear_span[0] + i*(x_shear_span[1] - x_shear_span[0])/ (step_per_dir-1) for i in range(step_per_dir)]
  y_shear = [y_shear_span[0] + i*(y_shear_span[1] - y_shear_span[0])/ (step_per_dir-1) for i in range(step_per_dir)]
  x_steps = [(x_span[0] + np.round(i*(x_span[1] - x_span[0])/ (step_per_dir-1))).astype('int') for i in range(step_per_dir)]
  y_steps = [(y_span[0] + np.round(i*(y_span[1] - y_span[0])/ (step_per_dir-1))).astype('int') for i in range(step_per_dir)]
  para =[]
  res = []
  near = []

  # for loops for all parameter combinations

  for xsh in x_shear:
    for ysh in y_shear:

      # if shear x and y are defined
      # calculate the sheared image

      M = np.float32([[1, xsh, 0],
             	[ysh, 1  , 0],
            	[0, 0  , 1]])          
      sheared_img = cv2.warpPerspective(image,M,image.shape)
      for xst in x_steps:
        for yst in y_steps:

          # calculate MSE for all images in the list 

          tmp = [ ((sheared_img[(63 + yst):(189 + yst),  (63 + xst):(189 + xst)
                               ].astype('float32')  - img_comp[63:189, 63: 189].astype(
                                   'float32') )**2).mean() for img_comp in  img_List_Comp]

          near.append(np.argmin(np.array(tmp)))
          res.append(min(tmp))
          para.append([xst, yst, xsh, ysh])
    ind = np.argmin(np.array(res))
  return([para[ind], res[ind], near[ind]])

def orientateImage(image, x_shift, y_shift, x_shear, y_shear):
  M = np.float32([[1, x_shear, 0],
             	[y_shear, 1  , 0],
            	[0, 0  , 1]])          
  sheared_img = cv2.warpPerspective(image,M,image.shape)
  res = np.zeros((image.shape[0] + 2*np.abs(y_shift), image.shape[1] + 2*np.abs(x_shift)))

  res[(np.abs(y_shift) - y_shift) :  (image.shape[0] + np.abs(y_shift) - y_shift), 
      (np.abs(x_shift) - x_shift) :  (image.shape[1] + np.abs(x_shift) - x_shift)
      ] = sheared_img

  return(res[
             np.abs(y_shift):(image.shape[0] + np.abs(y_shift)),
              np.abs(x_shift):(image.shape[1] + np.abs(x_shift)),
  ])

"""## Image Reconstruction algorithms

There are several ways of reconstruct the images based on principal component analysis. The following 4 ideas will be implemented:

- Complete: One image is one observation, each pixel one feature
- Rowwise: Each Row is one observation each pixel in that row a feature
- Colwise: Each Column is on observation and each pixel in that column is a feature
- Subimage: The images are divided in smaler frames and each frame is one observation, each pixel in that frame is one feature. 

The reconstruction power of the pca is the crucial point of the whole method: If the reconstruction power is too small the image reconstruction is very bad and the reconstruction error is to noisy to get any usefull information out of it. If the reconstruction power is to high, the pca will also reconstruct the anomlies.

### Complete Image Reconstruction

This method treat each image as one observation and each of the pixel as one feature. Each of the estimated principal components is therefore of size 2016*2016 and every pixel in the image is seen to be dependend of all other pixels. This is very memory intensiv and slow!
Implementation is on the other hand streight foreward.
"""

class comRecon:
  def __init__(self, n_components):
    self.pca = PCA(n_components)

  def fit(self, trainArray):
    n, col, row = trainArray.shape
    self.pca.fit(trainArray.reshape((n, row * col)))

  def reconstruct(self, image):
    row, col = image.shape
    recon = self.pca.inverse_transform(
        self.pca.transform(image.reshape((1, row * col)))
    ).reshape((row, col))
    recon = np.round(np.minimum(np.maximum(recon, 0), 255))
    return( recon )

"""## Rowwise Image Reconstruction

In this method the rows are seen as different observations and the columns as there features. By the straight forward implementation which would be to train one PCA for the whole images, we face quite quickly the problem, that we need many components to reconstruct the image precisly enough. But by using so many components (~100-200) the PCA will also be able to reconstruct the anomalys. So for that the following class will reconstruct the images based of divide the rows of  the image in subgroups for which reconstruction only a few components are enough. 

For get a better generalization in the fitting procedure also the rows of the subgroups in the neighbourhood are used. 
"""

# Class for reconstruct images rowwise

# contains the following methods:

# __init__(n_subgroups, n_somponents)
# n_subgroups: integer; defines in how many subgroups the rows are devided. 
#               For each subgroup a PCA is computed. 
#               should be an integer divisor of the row numbers of the image
# n_components: integer; defines in how many PCA components are used for reconstruction
# return: Nothing

# fit(trainArray) 
# trainArray: 3-D numpy-array [n x row x cols]
# return: Nothing

# reconstruct(image)
# image: [row x col]
# return: np.array [row x col] 

class rowRecon:
  def __init__(self, n_subgroups, n_components):
      self.n_subgroups = n_subgroups
      self.n_components =n_components
      self.pca_row = []

  # nsamples x rows x columns
  
  def fit(self, trainArray):
    self.nx = trainArray.shape[1]

    # For each subgroup a pca is computed
    # The fit is done with the rows of the subgroup and the rows of the subgroups
    # in the neighbourhood. This should make the reconstruction better in generalization

    for i in range(self.n_subgroups):
      self.pca_row.append(
          PCA(self.n_components).fit(np.concatenate(
            trainArray[:, max(i-2, 0)*( self.nx //self.n_subgroups) : min(i+3,self.n_subgroups +1 )*( self.nx //self.n_subgroups), :   ],
             axis = 0 )
          )
          )
      
  def reconstruct(self, image):

    # The recinstuction of the image is done by 
    # 1. divide the image in subgroups
    # 2. transform the rows with the pca of the subgroup
    # 3. inverse transform the result back in the space of image
    # 4. bound the reconstruction by [0, 255] and return uint8 matrix

    recon = np.zeros(image.shape)

    for i in range(self.n_subgroups):
      recon_row_tmp = self.pca_row[i].inverse_transform(
          self.pca_row [i].transform(
              image[
                    (i)*( self.nx // self.n_subgroups) : (i+1)*( self.nx // self.n_subgroups),
                    :  ]))
      recon[(i)*( self.nx // self.n_subgroups) : (i+1)*( self.nx // self.n_subgroups), : ] = recon_row_tmp
    
    recon = np.round(np.minimum(np.maximum(recon, 0), 255))
    return(recon)

"""## Colwise Image Reconstruction

Same like above for the transposed images. 
"""

# Just a wrapper for the function above

class colRecon:
  def __init__(self, n_subgroups, n_components):
      self.n_subgroups = n_subgroups
      self.n_components =n_components
      self.col_pca = rowRecon(self.n_subgroups, self.n_components)

  def fit(self, trainArray):
    self.col_pca.fit(np.swapaxes(trainArray, 1, 2))
  
  def reconstruct(self, image):
    return(
        np.swapaxes(self.col_pca.reconstruct(np.swapaxes(image, 0, 1)), 0,1)
    )

"""## Subimage Reconstruction

This method will use pca forsubframes of the images and afterwards it will put them back together. So the image is divided in n times n subframes and for each subframe a pca is fitted. 

In the fitting the pca of each subframe will also get some shifted version of the image - for better generalization. 
"""

from numpy.core.multiarray import concatenate
class subRecon:
  def __init__(self, n_subgroups, n_components):
      self.n_subgroups_x = n_subgroups
      self.n_subgroups_y = n_subgroups
      self.n_components =n_components
      self.pca_subimage = []

  def fit(self, trainArray):

    self.nx = trainArray.shape[1]
    self.ny = trainArray.shape[2]
    n_obs = trainArray.shape[0]
    g_x = self.nx //self.n_subgroups_x
    g_y = self.ny //self.n_subgroups_y

    # For all subgroups x
      # For all subgroups y
       # The original frame and overlapping neighbours are used to calculate 
       # the principal decomposition

    for i in range(self.n_subgroups_x):
      for j in range(self.n_subgroups_y):

        array_list = []

        array_list.append( trainArray[:, 
                          j*g_y : (j+1)*g_y, 
                          i*g_x : (i+1)*g_x   
                          ].reshape((n_obs, g_y * g_x)
                          ).copy())
        
        # now follow all overlapping neighbourhoods
        
        if (i > 0):
          array_list.append( trainArray[:, 
                          (j*g_y) : ((j+1)*g_y ), 
                          (i*g_x - g_x//2) : ((i+1)*g_x - g_x//2)
                          ].reshape((n_obs, g_y * g_x)
                          ).copy())
        if (j > 0) & (i > 0):
            array_list.append( trainArray[:, 
                          (j*g_y - g_y//2) : ((j+1)*g_y -  g_y//2), 
                          (i*g_x - g_x//2) : ((i+1)*g_x - g_x//2)
                          ].reshape((n_obs, g_y * g_x)
                          ).copy())
        if j != 0:
            array_list.append( trainArray[:, 
                          (j*g_y - g_y//2) : ((j+1)*g_y -  g_y//2), 
                          (i*g_x ) : ((i+1)*g_x)
                          ].reshape((n_obs, g_y * g_x)
                          ).copy())
        if i != (self.n_subgroups_x - 1):
          array_list.append( trainArray[:, 
                          (j*g_y) : ((j+1)*g_y ), 
                          (i*g_x + g_x//2) : ((i+1)*g_x + g_x//2)
                          ].reshape((n_obs, g_y * g_x)
                          ).copy())
        if (j != (self.n_subgroups_y - 1)) & (i != (self.n_subgroups_x - 1)):
            array_list.append( trainArray[:, 
                          (j*g_y + g_y//2) : ((j+1)*g_y +  g_y//2), 
                          (i*g_x + g_x//2) : ((i+1)*g_x + g_x//2)
                          ].reshape((n_obs, g_y * g_x)
                          ).copy())
        if j != (self.n_subgroups_y - 1):
            array_list.append( trainArray[:, 
                          (j*g_y + g_y//2) : ((j+1)*g_y +  g_y//2), 
                          (i*g_x ) : ((i+1)*g_x)
                          ].reshape((n_obs, g_y * g_x)
                          ).copy())
        if (j != (self.n_subgroups_y - 1)) & (i > 0):
            array_list.append( trainArray[:, 
                          (j*g_y + g_y//2) : ((j+1)*g_y +  g_y//2), 
                          (i*g_x - g_x//2) : ((i+1)*g_x - g_x//2)
                          ].reshape((n_obs, g_y * g_x)
                          ).copy())
        if (j > 0) & (i != (self.n_subgroups_x - 1)):
            array_list.append( trainArray[:, 
                          (j*g_y - g_y//2) : ((j+1)*g_y -  g_y//2), 
                          (i*g_x + g_x//2) : ((i+1)*g_x + g_x//2)
                          ].reshape((n_obs, g_y * g_x)
                          ).copy())

        
        self.pca_subimage.append(
            PCA(self.n_components).fit(
                np.concatenate(array_list, axis = 0)
            )
            )
  
  def reconstruct(self, image):
    recon = np.zeros(image.shape)

    for i in range(self.n_subgroups_x):
      for j in range(self.n_subgroups_y):
        recon[j*( self.ny //self.n_subgroups_y) : (j+1)*( self.ny//self.n_subgroups_y), 
              i*( self.nx //self.n_subgroups_x) : (i+1)*( self.nx//self.n_subgroups_x)] =  self.pca_subimage[i * self.n_subgroups_y + j].inverse_transform(
                  self.pca_subimage[i * self.n_subgroups_y + j].transform(
                      image[j*( self.ny //self.n_subgroups_y) : (j+1)*( self.ny//self.n_subgroups_y), 
                            i*( self.nx //self.n_subgroups_x) : (i+1)*( self.nx//self.n_subgroups_x)   ].reshape(
                            ( self.ny//self.n_subgroups_y * self.nx//self.n_subgroups_x)).reshape(1, -1)
            )
            ).reshape((self.ny//self.n_subgroups_y, self.nx//self.n_subgroups_x))
    recon = np.round(np.minimum(np.maximum(recon, 0), 255))
    return(recon)

"""## Wrapper Class with filter methods

This class has the only purpose to merge the reconstruction methods with some appropriate filter methods:

- all algorithms will calculate in train procedure also an Standard error Map
- getErrors method returns the reconstruction errors over a defined threshold
- getStadardErrors returns the standardized errors over a defined threshold
- getFilteredErrors return a binary error map based on some aditional filterin of the standard errors
  - the standard error map gets opend 3 times. That leads to an ommitting of errors which arent at least 7 pixels in each direction. Only the errors which survive this procedure are considered in the following filter step.
  - the biggest distance of 2 points in each error cluster is calculated. The ratio of the number of errorpoints and the square of the istance is calculated. Based on the assuption, that solderballs tend to be more compact square or circle shaped - the number of points in one error cluster should scale quadratic with the longest distance of to points. If the ratio of error points and the squared distence is below 1/3 the cluster is not considered to be an error. 
  - the standard error map gets opend 3 times. The errors which are caused by a shift in the structure will grow together an form a big error. For each left possible error cluster the growth of the error cluster is calculated and if thats above a threshold the cluster is not considered  to be an error any more. 

"""

from scipy.spatial.distance import pdist
class pcaRecon:
  def __init__(self, type, n_subgroups, n_components):

    if type == 'com':
      self.reconstructer = comRecon(n_components)

    elif type == 'row':
      self.reconstructer  = rowRecon(n_subgroups, n_components)

    elif type == 'col':
      self.reconstructer  = colRecon(n_subgroups, n_components)

    elif type == 'sub':
      self.reconstructer  = subRecon(n_subgroups, n_components)
  
  def fit(self, trainArray): 
    self.reconstructer.fit(trainArray)
    self.SEMap = np.zeros((trainArray.shape[1], trainArray.shape[2]))
    for i in range(trainArray.shape[0]):
      self.SEMap = self.SEMap  + np.abs(trainArray[i, :, :] - self.reconstruct(trainArray[i,:,:])) /trainArray.shape[0] 
 
  def reconstruct(self, image): 
    return(self.reconstructer.reconstruct(image))

  def getErrors(self, image, raw_threshold):
    errors = self.reconstruct(image) - image
    errors[errors < raw_threshold] = 0
    return(errors)

  def getStandardErrors(self, image, raw_threshold, SE_threshold):
    errors = 5 * self.getErrors(image, raw_threshold) /np.maximum(self.SEMap, 5) 
    errors[errors < SE_threshold] = 0
    return(errors)

  def getFilteredErrors(self, image, raw_threshold, SE_threshold, max_errGrow):
    errorMap = self.getStandardErrors(image, raw_threshold, SE_threshold)
    errorMap[errorMap > 0 ] = 1

    label_map = measure.label(errorMap, background=0)
    filteredErrorMap = np.zeros((2016, 2016))

    footprint = disk(3)
    closed_error_map = morphology.binary_closing(image = errorMap, selem = footprint)
    label_map_closed = measure.label(closed_error_map, background=0)

    opened_error_map = morphology.binary_opening(image = errorMap, selem = footprint)
    label_map_opened = measure.label(opened_error_map, background=0)

    for i in range(label_map_opened.max()):

      # For all Groups of errors in the opened map 
      # Calculate the size 
      # Then calculculate the size of the clusters in the closed image
      # which the opend error belong to

      cnt_opend = np.sum(label_map_opened == (i + 1))
      closed_labels = np.unique(label_map_closed[label_map_opened == (i + 1)])
      cnt_closed = 0

      data_matrix = np.argwhere(label_map_opened == (i + 1))
      x_max = data_matrix[:, 0].argmax()
      x_min = data_matrix[:, 0].argmin()
      y_max = data_matrix[:, 1].argmax()
      y_min = data_matrix[:, 1].argmin()

      max_dist = np.max(pdist(data_matrix[ [x_max, x_min, y_max, y_min] , :]))

      for j in closed_labels:
        if j > 0:
          cnt_closed = cnt_closed + np.sum(label_map_closed == j)


      if ( 
          ((cnt_closed / cnt_opend) < max_errGrow) &  
          # error dont grow
          ((max_dist**2 )/4 < cnt_opend )
          # errors are compact 
          ):
          filteredErrorMap[label_map_opened == (i+1)] = 1

    return(filteredErrorMap)

"""Good running parameter setups:
- com: not evaluatet in last days
- col / row: threshold_raw = 20, threshold_SE = 20, err_grow = 5
- sub: threshold_raw = 10, threshold_SE = 10, err_grow = 5

## Final Anomaly Detector

For the final anomaly detection multiple reconstruction methods are used. Each of the reconstruction methods leads to different errormaps. All should include the true anomalies. Because each method uses an other representation of the data, the errors which occor, False anomalies, should be different. By using several reconstruction algorithm and define an error only then as error if all methods say its an error.

The following improvement steps are to go:
- include the structure growth filter from the other document
- include an orientation preprocess
- implemented but not checked
- include an heatmap plot
"""

class anomalyDetector:
  def __init__ (self, type_list, n_subgroups_list, n_components_list, raw_th_list,  SE_th_list, err_grow_list, optimize_orientation = False):
    self.reconstructer = []
    self.raw_th_list = raw_th_list
    self.SE_th_list = SE_th_list
    self.err_grow_list = err_grow_list
    self.optim_flag = optimize_orientation

    if optimize_orientation:
      self.train_list_small = []
      self.train_list = []


    for i in range(len(type_list)):
      self.reconstructer.append(pcaRecon(type_list[i], n_subgroups_list[i], n_components_list[i]))
  
  def fit(self, trainArray):
    if self.optim_flag:
      for i in range(trainArray.shape[0]):
        self.train_list_small.append(cv2.resize(trainArray[i, :, :], (252, 252), interpolation = cv2.INTER_AREA))
        self.train_list.append(trainArray[i, :, :])

    for i in range(len(self.reconstructer)):
      self.reconstructer[i].fit(trainArray)

  def getFilteredErrors(self, image):
    if self.optim_flag:
      paras, res, neigh = findBestOrientation(cv2.resize(image, (252, 252), interpolation = cv2.INTER_AREA), self.train_list_small,
                          [-20, 20], [-20, 20], [-0.02, 0.02], [-0.02, 0.02], 5)
      tmp_image = orientateImage(image, 8*paras[0], 8*paras[1], paras[2], paras[3])
      im_ind = orientateImage(np.ones(image.shape), 8*paras[0], 8*paras[1], paras[2], paras[3])
      tmp_neigh = self.train_list[neigh]
      tmp_image[im_ind < 1] = tmp_neigh[im_ind < 1] 
    else: 
      tmp_image = image

    votes = np.zeros(image.shape)
    for i in range(len(self.reconstructer)):
      votes += self.reconstructer[i].getFilteredErrors(
          image, self.raw_th_list[i], self.SE_th_list[i], self.err_grow_list[i])
    
    if self.optim_flag:
      rev_ind = orientateImage(np.ones(image.shape), -8*paras[0], -8*paras[1], -paras[2], -paras[3])
      votes = orientateImage(votes, -8*paras[0], -8*paras[1], -paras[2], -paras[3])
      votes[rev_ind < 1] = 0
      
    # here the structure grow prcedurehas to be 

    return(votes == len(self.reconstructer))

  def detect(self, image):
    errorMap = self.getFilteredErrors(image)
    return([errorMap.sum() > 0, errorMap])

"""# Heapmap Visualization for Bosh"""

def heatmap(image, error):
  #plt.imshow(error, cmap = 'gray')
  #plt.imsave("../Ablage/tmp_err.png")
  #err  = cv.imread("../Ablage/tmp_err.png").astype(np.float32)

  (x,y)=err.shape
  hmap = np.empty((x,y,4),np.float32)

  hmap[np.where((err==[True]))] = [1,0,0,1]
  hmap[np.where((err!=[True]))] = [1,1,1,0]
  
  fig, ax = plt.subplots()
  ax.imshow(image,cmap = 'gray')
  ax.imshow(hmap, alpha = 0.5)
  return(fig)

#Run)
aD = anomalyDetector(['sub', 'col', 'row'], [8, 48, 48], 
                     [50, 35, 50], [20, 10, 10], 
                     [10, 10, 10], [5, 5, 10], False)
aD.fit(np.array(image_list_train))

counter = 0
for elem in image_list_train:
  img= image_list_test[counter]
  err= aD.detect(image_list_test[counter])[1]
  heatmap(img, err)
  plt.savefig("../Results/heatmaps/01.28/hmap_"+str(counter)+".png", transparent=True)
  counter +=1
