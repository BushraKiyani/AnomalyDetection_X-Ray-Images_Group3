import numpy as np
import cv2 as cv2
from sklearn.decomposition import PCA
from skimage import measure
from skimage import data, filters, color, morphology
from skimage.morphology import disk

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
          ((cnt_closed / cnt_opend) < max_errGrow) and  
          # error dont grow
          ((max_dist**2 )/4 < cnt_opend )
          # errors are compact 
          ):
          filteredErrorMap[label_map_opened == (i+1)] = 1

    return(filteredErrorMap)    



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

  def getFilteredErrorsPlus(self, image):
    votes = np.zeros(image.shape)
    recon_images = []
    for i in range(len(self.reconstructer)):
      tmp = self.reconstructer[i].getFilteredErrors(
          image, self.raw_th_list[i], self.SE_th_list[i], self.err_grow_list[i])
      recon_images.append(tmp)
      votes += tmp

    footprint = disk(3)
    closed_error_map = morphology.binary_closing(image = (votes > 0), selem = footprint)
    label_map_closed = measure.label(closed_error_map, background=0)
    error_map = np.zeros((2016, 2016))
    for i in range(label_map_closed.max()):
      if i > 0:
        v = 0 
        for j in range(len(self.reconstructer)):
          if (recon_images[j][label_map_closed == i]).sum() > 0:
            v += 1
        if (v == len(self.reconstructer)):
          error_map[label_map_closed == i] = 1
    return(error_map)
    


  def detect(self, image):
    errorMap = self.getFilteredErrors(image)
    return([errorMap.sum() > 0, errorMap])      
