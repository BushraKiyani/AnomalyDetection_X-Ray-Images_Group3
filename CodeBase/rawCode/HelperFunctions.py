import numpy as np
import cv2 as cv2
from sklearn.decomposition import PCA
from skimage import measure
from skimage import data, filters, color, morphology
from skimage.morphology import disk
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


def heatmap(image, error):
    img_shape = error.shape
    if len(img_shape) == 2:
        tmp_img =  cv2.cvtColor(error.astype('uint8'), cv2.COLOR_GRAY2RGB) 
    else:
        tmp_img = error.astype('uint8')
    
    x, y, c = tmp_img.shape
    hmap = np.empty((x,y,c+1),np.float32)
    hmap[np.where((error!=[0,0,0]).all(axis=2))] = [255,0,0,0.5]
    
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(hmap)
    return(fig)

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


def create_artificial_anomaly(img, diff = 80):
  kernel = np.ones((8,8),np.uint8)
  poss_seed_pnts = np.where(cv2.erode((img > 120).astype('uint8'), kernel, 1))
  poss_seed_pnts =np.array([poss_seed_pnts[0], poss_seed_pnts[1]]).transpose()
  idx_row = (poss_seed_pnts[:, 0] > 500) & (poss_seed_pnts[:, 0] < 1250)
  idx = np.random.choice(range(poss_seed_pnts[idx_row, :].shape[0]), 1)
  seed_pnt = poss_seed_pnts[idx_row, :][idx]

  n = np.random.binomial(20, .4)
  if n > 0:
    seed_pnts = np.stack( [
        np.random.multivariate_normal(
            seed_pnt[0], np.diag([16, 16]), n
            ).astype('int')[0],
        seed_pnt[0]],
        axis = 0 )
  else:
    seed_pnts = seed_pnt[0]
  
  seed_pnts = np.maximum(seed_pnts, 0)
  seed_pnts = np.minimum(seed_pnts, 2015)  

  n_close = np.random.binomial(16, .5)
  error_map = np.zeros((2016, 2016)) 
  error_map[seed_pnts[:, 0], seed_pnts[:, 1]] = 1
  error_map = error_map.astype('uint8')

  kernel = np.ones((n_close, n_close),np.uint8)
  error_map = cv2.dilate(error_map, kernel, 1)
  # background_add = img[error_map > 0].mean() / 4 
  # darkness = np.round(np.random.standard_normal(1) * 20 + diff + background_add)
  darkness = diff 
  art_err_img = img - darkness * error_map
  return( (art_err_img, error_map) )


def k_fold_art_cv(recon_method, folds, image_array, subgroups, n_components, n_art = 5):
  n, h, w = image_array.shape
  fold_group = np.random.choice(np.array(range(n)) % folds, n, replace = False)
  q05_nok = []
  g05_nok = []
  mean_nok = []
  for i in range(folds):
    idx = np.where(fold_group == i)[0]
    recon = pcaRecon(recon_method, subgroups, n_components)
    recon.fit(image_array[fold_group != i])
    for j in range(len(idx)):
      for k in range(n_art):
        art_an, err_map = create_artificial_anomaly(image_array[idx[j] ,: ,:].astype('uint8'), 50 )
        recon_img = recon.reconstruct(art_an)
        error = recon_img - art_an
        q05 = np.quantile(error[err_map > 0], .05)
        g05 = np.mean(error[err_map == 0] > q05)
        mean_nok.append(error[err_map > 0] )
        q05_nok.append(q05)
        g05_nok.append(g05)
  return([np.quantile(np.array(g05_nok), .5), q05_nok, g05_nok])

def k_fold_para_selection(recon_method, folds, image_array, subgroups, n_components, n_art = 5):
    subgroups_per_iter  = []
    components_per_iter  = []
    cv_score = []
    
    for sg in subgroups:
        for nc in n_components:
            cv_res =  k_fold_art_cv(recon_method, folds, image_array, sg, nc, n_art)
            subgroups_per_iter.append(sg)
            components_per_iter.append(nc)
            cv_score.append(cv_res[0])
    idx_min = np.argmin(cv_score)
    best_sg = subgroups_per_iter[idx_min]
    best_nc = components_per_iter[idx_min]
    return([(best_sg, best_nc), (cv_score, subgroups_per_iter, components_per_iter)])


''' def k_fold_threshold_cv(recon_method, 
                        folds, 
                        image_array, 
                        subgroups, 
                        n_components,
                        err_thresholds,
                        # std_err_thresholds,
                        first_closing,
                        first_opening,
                        second_closing,
                        second_opening, 
                        n_art = 5):
    n, h, w = image_array.shape
    fold_group = np.random.choice(np.array(range(n)) % folds, n, replace = False)
    th_para = []
    fo_para = []
    fc_para = []
    so_para = []
    sc_para = []
    cv_score = []
    anomaly_prct = []
    total_res = []
    for i in range(folds):
        idx = np.where(fold_group == i)[0]
        recon = pcaRecon(recon_method, subgroups, n_components)
        recon.fit(image_array[fold_group != i])
        for j in range(len(idx)):
            print(j)
            for k in range(n_art):
                print('n_art')
                art_an, err_map = create_artificial_anomaly(image_array[idx[j] ,: ,:].astype('uint8'), 50 )
                recon_img = recon.reconstruct(art_an)
                recon_error = recon_img - art_an
                for et in err_thresholds:
                    print('et')
                    err_map_th = (recon_error > et)
                    for fc in first_closing:
                        print('fc')
                        footprint = disk(fc)
                        fc_error_map = morphology.binary_closing(image = err_map_th, footprint = footprint)
                        for fo in first_opening:
                            print('fo')
                            footprint = disk(fo)
                            fo_error_map = morphology.binary_opening(image = fc_error_map, footprint = footprint)
                            for sc in second_closing:
                                print('sc')
                                footprint = disk(fc)
                                sc_error_map = morphology.binary_closing(image = fo_error_map, footprint = footprint)
                                for so in second_opening:
                                    print('so')
                                    footprint = disk(so)
                                    so_error_map = morphology.binary_opening(image = sc_error_map, footprint = footprint)
                                    th_para.append(et)
                                    fo_para.append(fo)
                                    fc_para.append(fc)
                                    so_para.append(so)
                                    sc_para.append(sc)
                                    total = so_error_map.sum()
                                    anomaly_sum = so_error_map[err_map > 0].sum()
                                    cv_score.append(anomaly_sum / total)
                                    anomaly_prct.append(so_error_map[err_map > 0].mean())
                                    total_res.append(total)
    return([cv_score, anomaly_prct, total_res]) '''

def k_fold_threshold_cv(recon_method, 
                        folds, 
                        image_array, 
                        subgroups, 
                        n_components,
                        err_thresholds,
                        # std_err_thresholds,
                        first_closing,
                        first_opening,
                        n_art = 5):
    n, h, w = image_array.shape
    fold_group = np.random.choice(np.array(range(n)) % folds, n, replace = False)
    th_para = []
    fo_para = []
    fc_para = []
    cv_score = []
    anomaly_prct = []
    total_res = []
    for i in range(folds):
        idx = np.where(fold_group == i)[0]
        recon = pcaRecon(recon_method, subgroups, n_components)
        recon.fit(image_array[fold_group != i])
        for j in range(len(idx)):
            print((i * n/folds + j)/n )
            for k in range(n_art):
                art_an, err_map = create_artificial_anomaly(image_array[idx[j] ,: ,:].astype('uint8'), 50 )
                recon_img = recon.reconstruct(art_an)
                recon_error = recon_img - art_an
                for et in err_thresholds:
                    err_map_th = (recon_error > et)
                    
                    for fc in first_closing:
                        footprint = disk(fc)
                        fc_error_map = morphology.binary_closing(image = err_map_th, footprint = footprint)
                        for fo in first_opening:
                            footprint = disk(fo)
                            fo_error_map = morphology.binary_opening(image = fc_error_map, footprint = footprint)
                            th_para.append(et)
                            fo_para.append(fo)
                            fc_para.append(fc)
                            total = fo_error_map.sum()
                            anomaly_sum = fo_error_map[err_map > 0].sum()
                            cv_score.append(anomaly_sum / total)
                            anomaly_prct.append(fo_error_map[err_map > 0].mean())
                            total_res.append(total)
    return([cv_score, th_para, fo_para, fc_para, anomaly_prct, total_res])


# def k_fold_threshold_cv(recon_method, 
#                         folds, 
#                         image_array, 
#                         subgroups, 
#                         n_components,
#                         err_thresholds,
#                         # std_err_thresholds,
#                         first_closing,
#                         first_opening,
#                         n_art = 5):
#     n, h, w = image_array.shape
#     fold_group = np.random.choice(np.array(range(n)) % folds, n, replace = False)
#     th_para = []
#     fo_para = []
#     fc_para = []
#     cv_score = []
#     anomaly_prct = []
#     total_res = []
#     for i in range(folds):
#         idx = np.where(fold_group == i)[0]
#         recon = pcaRecon(recon_method, subgroups, n_components)
#         recon.fit(image_array[fold_group != i])
#         for j in range(len(idx)):
#             print((i * n/folds + j)/n )
#             for k in range(n_art):
#                 art_an, err_map = create_artificial_anomaly(image_array[idx[j] ,: ,:].astype('uint8'), 50 )
#                 recon_img = recon.reconstruct(art_an)
#                 recon_error = recon_img - art_an
#                 for et in err_thresholds:
#                     err_map_th = (recon_error > et)
#                     for fo in first_opening:
#                         footprint = disk(fo)
#                         fo_error_map = morphology.binary_opening(image =  err_map_th, footprint = footprint)
#                         for fc in first_closing:
#                             footprint = disk(fc)
#                             fc_error_map = morphology.binary_closing(image = fo_error_map, footprint = footprint)
#                             th_para.append(et)
#                             fo_para.append(fo)
#                             fc_para.append(fc)
#                             total = fc_error_map.sum()
#                             anomaly_sum = fc_error_map[err_map > 0].sum()
#                             cv_score.append(anomaly_sum / total)
#                             anomaly_prct.append(fc_error_map[err_map > 0].mean())
#                             total_res.append(total)
#     return([cv_score, th_para, fo_para, fc_para, anomaly_prct, total_res])
                      
def evaluate_error_map(bin_err_map, row_min, row_max, col_min, col_max):
      cnt_err_window = bin_err_map[row_min:row_max, col_min:col_max]
      hit_window = (cnt_err_window.sum() > 15)
      labels_map = measure.label(bin_err_map, background=0)
      no_err_cluster = np.max(labels_map)
      return([hit_window, no_err_cluster])
      
      
def show_error(img, rmin, rmax, cmin, cmax):
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(img, cmap = 'gray')
    # Create a Rectangle patch
    rect = patches.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()
