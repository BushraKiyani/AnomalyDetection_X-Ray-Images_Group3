import numpy as np
import cv2 as cv2
from sklearn.decomposition import PCA
from skimage import measure
from skimage import data, filters, color, morphology
from skimage.morphology import disk


class quantile_filter:
    
    
    def __init__(self, quantile):
        self.quantile = quantile
    
    def reconstruct_fit(self, reconstructer, image_array):
        self.reconstrucer = reconstructer
        
        recon_array = np.array(
            [reconstructer.reconstruct(image_array[i, :, :]) for i in range(image_array.shape[0])]
        )
        err_arr = image_array - recon_array
        self.quantile_map = np.quantile(err_arr, self.quantile, axis = 0)
        
    def fit(self, recon_array, image_array):
        err_arr = image_array - recon_array
        self.quantile_map = np.quantile(err_arr, self.quantile, axis = 0)
        
    def reconstruct_filter(self, img):
        recon_img = self.reconstrucer.reconstruct(img)
        err_map = recon_img - recon_img
        ind_map = (err_map > self.quantile_map).astype('int')
        return(ind_map * err_map)
    
    def filter(self, err_map):
        ind_map = (err_map > self.quantile_map).astype('int')
        return(ind_map * err_map)
    
        
            
class mean_filter:
    
    
    def __init__(self):
        pass
    
    def reconstruct_fit(self, reconstructer, image_array):
        self.reconstrucer = reconstructer
    
        recon_array = np.array(
            [reconstructer.reconstruct(image_array[i, :, :]) for i in range(image_array.shape[0])]
        )
        err_arr = image_array - recon_array
        self.mean_map = np.mean(err_arr, axis = 0)
        
    def fit(self, recon_array, image_array):
        err_arr = image_array - recon_array
        self.mean_map = np.quantile(err_arr, axis = 0)
        
    def reconstruct_filter(self, img):
        recon_img = self.reconstrucer.reconstruct(img)
        err_map = recon_img - recon_img
        ind_map = (err_map > self.mean).astype('uint8')
        return(ind_map * err_map)
    
    def filter(self, err_map):
        ind_map = (err_map > self.mean).astype('uint8')
        return(ind_map * err_map)



class threshold_filter:
    
    def __init__(self, threshold):
        self.threshold = threshold
    
    def reconstruct_fit(self, reconstructer):
        self.reconstrucer = reconstructer
        
    def reconstruct_filter(self, img):
        recon_img = self.reconstrucer.reconstruct(img)
        err_map = recon_img - recon_img
        ind_map = (err_map > self.threshold).astype('uint8')
        return(ind_map * err_map)
    
    def filter(self, err_map):
        ind_map = (err_map > self.threshold).astype('uint8')
        return(ind_map * err_map)



class std_filter:
    
    def __init__(self, std_threshold, min_std):
        self.std_threshold = std_threshold
        self.min_std = min_std
    
    def reconstruct_fit(self, reconstructer, image_array):
        self.reconstrucer = reconstructer
    
        recon_array = np.array(
            [reconstructer.reconstruct(image_array[i, :, :]) for i in range(image_array.shape[0])]
        )
        err_arr = image_array - recon_array
        self.std_map = np.std(err_arr, axis = 0)
        self.std_map = np.maximum(self.std_map, self.min_std)
        
    def fit(self, recon_array, image_array):
        err_arr = image_array - recon_array
        self.std_map = np.std(err_arr, axis = 0)
        self.std_map = np.maximum(self.std_map, self.min_std)

        
        
    def reconstruct_filter(self, img):
        recon_img = self.reconstrucer.reconstruct(img)
        err_map = recon_img - recon_img
        ind_map = (err_map /self.std > self.std_threshold).astype('uint8')
        return(ind_map * err_map)
    
    def filter(self, err_map):
        ind_map = (err_map /self.std_map > self.std_threshold).astype('uint8')
        return(ind_map * err_map)



class morphological_filter:
    
    def __init__(self, kernel_size):
        self.kernel = disk(kernel_size)
    
    def filter(self, err_map):
        morph_err_map = cv2.morphologyEx(err_map, cv2.MORPH_OPEN, self.kernel)
        return(morph_err_map)
    
    
    
class errGrowth_filter:
    
    def __init__(self, kernel_size_open, kernel_size_close, errGrow_threshold):
        self.kernel_open = disk(kernel_size_open)
        self.kernel_close = disk(kernel_size_close)
        self.err_growth_threshold = errGrow_threshold
    
    
    def filter(self, err_map):
        closed_err_map = cv2.morphologyEx((err_map >0).astype('uint8') , cv2.MORPH_CLOSE, self.kernel_close)
        opened_err_map = cv2.morphologyEx((err_map > 0).astype('uint8'), cv2.MORPH_OPEN, self.kernel_open)
        opened_labels_map = measure.label((opened_err_map > 0).astype('uint8') , background=0)
        closed_labels_map = measure.label((closed_err_map > 0).astype('uint8') , background=0, connectivity = 2)
        ind_map = np.zeros(err_map.shape).astype('uint8')
             
        for oe in range(opened_labels_map.max() + 1):
            if oe != 0:
                cnt_op = np.sum(opened_labels_map == oe)
                label_opened = np.unique(closed_labels_map[opened_labels_map == oe])[0]
                cnt_cl = np.sum(closed_labels_map == label_opened)
                if (cnt_cl/cnt_op) < self.err_growth_threshold:
                    ind_map[opened_labels_map == oe] = 1
                          
        return(ind_map * err_map)
    


class augmented_merge_filter:
    def __init__(self, kernel_size):
        self.kernel = disk(kernel_size)
        
    def filter_merge(self, err_list):
        closed_array = np.array([cv2.morphologyEx(err.astype('uint8'), cv2.MORPH_CLOSE, self.kernel) for err in err_list])
        return(np.min(closed_array, axis = 0))
    
    

class augmented_merge_majority_filter:
    def __init__(self, kernel_size):
        self.kernel = disk(kernel_size)
        
    def filter_merge(self, err_list):
        closed_array = np.array([cv2.morphologyEx(err.astype('uint8'), cv2.MORPH_CLOSE, self.kernel) for err in err_list])
        return(np.sum(closed_array > 0, axis = 0) > (len(err_list) / 2))
    

