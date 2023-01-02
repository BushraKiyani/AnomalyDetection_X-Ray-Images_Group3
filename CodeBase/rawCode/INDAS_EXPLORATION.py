import numpy as np
import pandas as pd
import cv2 as cv
import cv2 as cv2
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib
from os import listdir
from os.path import isfile, join
import seaborn as sns

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Path loading 

path_ok = 'D:\\Indas\\OK_samples_xray\\OK_samples_xray/'
OK_path_001 = []
OK_path_002 = []
OK_path_003 = []
OK_path_004 = []

for subDir in listdir(path_ok):
    files = [f for f in listdir(join(path_ok, subDir))]
    cont_001_new = ['001_new' in f for f in files]
    cont_002_new = ['002_new' in f for f in files]
    cont_003_new = ['003_new' in f for f in files]
    cont_004_new = ['004_new' in f for f in files]   
    if np.any(cont_001_new):
         OK_path_001.append(path_ok + '/' + subDir + '/' + files[np.where(cont_001_new)[0][0]])
    if np.any(cont_002_new):
        OK_path_002.append(path_ok + '/' + subDir + '/' + files[np.where(cont_002_new)[0][0]])
    if np.any(cont_003_new):
        OK_path_003.append(path_ok + '/' + subDir + '/' + files[np.where(cont_003_new)[0][0]])
    if np.any(cont_004_new):
        OK_path_004.append(path_ok + '/' + subDir + '/' + files[np.where(cont_004_new)[0][0]])


path_nok = 'D:\\Indas\\NOK_samples_xray\\NOK_samples_xray/'
NOK_path_001 = []
NOK_path_002 = []
NOK_path_003 = []
NOK_path_004 = []

for subDir in listdir(path_nok):
    files = [f for f in listdir(join(path_nok, subDir))]
    cont_001_new = ['001_new' in f for f in files]
    cont_002_new = ['002_new' in f for f in files]
    cont_003_new = ['003_new' in f for f in files]
    cont_004_new = ['004_new' in f for f in files]   
    if np.any(cont_001_new):
         NOK_path_001.append(path_nok + '/' + subDir + '/' + files[np.where(cont_001_new)[0][0]])
    if np.any(cont_002_new):
        NOK_path_002.append(path_nok + '/' + subDir + '/' + files[np.where(cont_002_new)[0][0]])
    if np.any(cont_003_new):
        NOK_path_003.append(path_nok + '/' + subDir + '/' + files[np.where(cont_003_new)[0][0]])
    if np.any(cont_004_new):
        NOK_path_004.append(path_nok + '/' + subDir + '/' + files[np.where(cont_004_new)[0][0]])


same_date_neighbour_OK = [ ]
same_date_neighbour_NOK = [ ]
same_date_neighbour_ALL = [ ]


for angle in range(1, 5):

    # Data Loading 
    OK_PATH = [OK_path_001, OK_path_002, OK_path_003, OK_path_004][angle - 1] 
    NOK_PATH = [NOK_path_001, NOK_path_002, NOK_path_003, NOK_path_004][angle - 1]

    print(angle)
    img_arr_OK = np.array([cv2.imread(img_name, cv2.IMREAD_REDUCED_GRAYSCALE_8) for  img_name in OK_PATH]).astype('float32')
    print('OK LOADED')
    img_arr_NOK = np.array([cv2.imread(img_name, cv2.IMREAD_REDUCED_GRAYSCALE_8) for  img_name in  NOK_PATH]).astype('float32')
    print('NOK LOADED')


    ##########################################################################################################
    #################################### MEAN AND SD PLOTS  ##################################################
    ##########################################################################################################

    matplotlib.rcParams.update({'font.size': 22})

    fig, ax = plt.subplots(2,3, figsize=(15,10))
    #fig.suptitle('')
    im1 = ax[0, 0].imshow(img_arr_OK.mean(axis = 0), cmap='gray')
    ax[0, 0].axis('off')
    im2 = ax[1, 0].imshow(img_arr_OK.std(axis = 0), cmap = 'gray')
    ax[1, 0].axis('off')
    im1 =  ax[0, 2].imshow(img_arr_NOK.mean(axis = 0), cmap = 'gray')
    ax[0, 2].axis('off')
    im2 = ax[1, 2].imshow(img_arr_NOK.std(axis = 0), cmap = 'gray')
    ax[1, 2].axis('off')
    im1 =  ax[0, 1].imshow(np.concatenate([img_arr_OK, img_arr_NOK]).mean(axis = 0), cmap='gray')
    ax[0, 1].axis('off')
    im2 = ax[1, 1].imshow(np.concatenate([img_arr_OK, img_arr_NOK]).std(axis = 0), cmap = 'gray')
    ax[1, 1].axis('off')
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.8,
                        wspace=0.2, hspace=0.1)
    cb_ax1 = fig.add_axes([0.83, 0.6, 0.02, 0.25])
    cbar1 = fig.colorbar(im1, cax=cb_ax1)
    cb_ax2 = fig.add_axes([0.83, 0.25, 0.02, 0.25])
    cbar2 = fig.colorbar(im2, cax=cb_ax2)

    fig.suptitle('Mean and SD Images by Category Angle ' + str(angle), y = 0.95)
    fig.supxlabel('OK                           All                            NOK         ', y = 0.15)
    fig.supylabel('         SD                          Mean', x = 0.05)

    plt.savefig('../plots\\mean_SD_images_angle' + str(angle) + '.png')
    plt.close()

    ###########################################################################################################
    ######################### PLOT PIXELSWISE STD PER ROW OK / ALL /NOK########################################
    ###########################################################################################################

    matplotlib.rcParams['figure.figsize'] = [15, 10]
    matplotlib.rcParams.update({'font.size': 22})

    fig = sns.lineplot(data = pd.DataFrame({'mean_std' :img_arr_OK.std(axis = 0).mean(axis = 1), 'row_number' : range(252)}),
                x = 'row_number', y = 'mean_std', color = 'green')
    sns.lineplot(data = pd.DataFrame({'mean_std' :np.concatenate([img_arr_NOK, img_arr_OK]).std(axis = 0).mean(axis = 1), 'row_number' : range(252)}),
                x = 'row_number', y = 'mean_std', color = 'blue')
    sns.lineplot(data = pd.DataFrame({'mean_std' :img_arr_NOK.std(axis = 0).mean(axis = 1), 'row_number' : range(252)}),
                x = 'row_number', y = 'mean_std', color = 'red')
    fig.set_xlabel("Row number")
    fig.set_ylabel("Mean pixelwise sd")
    fig.set_title('Comparison of pixelwise sd for Angle ' + str(angle))
    plt.legend(labels=["Normal Images","All Images", 'Anomal Images'])
    plt.savefig('../plots/comp_sd_pixelwise_per_row_angle' + str(angle) + '.png', pad_inches = 0)
    plt.close()

    ###########################################################################################################
    ######################### PLOT PIXELSWISE STD PER COL OK / ALL /NOK########################################
    ###########################################################################################################

    matplotlib.rcParams['figure.figsize'] = [15, 10]
    matplotlib.rcParams.update({'font.size': 22})

    fig = sns.lineplot(data = pd.DataFrame({'mean_std' :img_arr_OK.std(axis = 0).mean(axis = 0), 'row_number' : range(252)}),
                x = 'row_number', y = 'mean_std', color = 'green')
    sns.lineplot(data = pd.DataFrame({'mean_std' :np.concatenate([img_arr_NOK, img_arr_OK]).std(axis = 0).mean(axis = 0), 'row_number' : range(252)}),
                x = 'row_number', y = 'mean_std', color = 'blue')
    sns.lineplot(data = pd.DataFrame({'mean_std' :img_arr_NOK.std(axis = 0).mean(axis = 0), 'row_number' : range(252)}),
                x = 'row_number', y = 'mean_std', color = 'red')
    fig.set_xlabel("Column number")
    fig.set_ylabel("Mean pixelwise sd")
    fig.set_title('Comparison of pixelwise sd for Angle ' + str(angle))
    plt.legend(labels=["Normal Images","All Images", 'Anomal Images'])
    plt.savefig('../plots/comp_sd_pixelwise_per_col_angle' + str(angle) + '.png', pad_inches = 0)
    plt.close()

    ###########################################################################################################
    ############################## COMPUTE DISTANCES TO NEXT IMAGE  ###########################################
    ###########################################################################################################

    # For OK - Images 

    dist_mat_OK_OK = np.zeros((img_arr_OK.shape[0], img_arr_OK.shape[0]))
    for i in range(img_arr_OK.shape[0] - 1):
        dist_mat_OK_OK[(i+1): , i] = ((img_arr_OK[(i+1) : img_arr_OK.shape[0],:,:] - img_arr_OK[i])**2).sum(axis = (1,2))
    
    dist_mat_OK_OK = dist_mat_OK_OK + dist_mat_OK_OK.transpose()

    # Fill diagonal for easier searching
    np.fill_diagonal(dist_mat_OK_OK, 10*np.max(dist_mat_OK_OK))

    dist_mat_NOK_OK = np.zeros((img_arr_OK.shape[0], img_arr_NOK.shape[0]))
    for i in range(img_arr_NOK.shape[0]):
        dist_mat_NOK_OK[:, i] = ((img_arr_OK[:,:,:] - img_arr_NOK[i])**2).sum(axis = (1,2))
    
    dist_mat_NOK_NOK = np.zeros((img_arr_NOK.shape[0], img_arr_NOK.shape[0]))
    for i in range(img_arr_NOK.shape[0]):
        dist_mat_NOK_NOK[(i+1):, i] = ((img_arr_NOK[(i+1):,:,:] - img_arr_NOK[i])**2).sum(axis = (1,2))
    
    dist_mat_NOK_NOK = dist_mat_NOK_NOK + dist_mat_NOK_NOK.transpose()

    # Fill diagonal for easier searching
    np.fill_diagonal(dist_mat_NOK_NOK, 10*np.max(dist_mat_NOK_NOK))
  

    all_distances = np.concatenate(
      [np.concatenate([dist_mat_OK_OK, dist_mat_NOK_OK.transpose()]),
       np.concatenate([dist_mat_NOK_OK, dist_mat_NOK_NOK])], axis = 1)
    

    # Compare Dates with next neighbour

    dates_NOK = [str.split(str.split(f, '/')[3], '_')[1] for f in  NOK_PATH]
    dates_OK = [str.split(str.split(f, '/')[3], '_')[1] for f in  OK_PATH]
    all_dates = dates_OK + dates_NOK

    same_date_neighbour_OK .append(
        np.mean(
            [dates_OK[dist_mat_OK_OK.argmin(axis = 0)[i]] == dates_OK[i] for i in range(len(dates_OK))]
            )
        )
    same_date_neighbour_NOK.append(
        np.mean(
            [dates_OK[dist_mat_NOK_NOK.argmin(axis = 0)[i]] == dates_NOK[i] for i in range(len(dates_NOK))]
            )
        )
    same_date_neighbour_ALL.append(
        np.mean(
            [
                (dates_OK + dates_NOK)[all_distances.argmin(axis = 0)[i]] == (dates_OK + dates_NOK)[i] 
                for i in range(len(dates_OK + dates_NOK))
                ]
            ))
    

    ###########################################################################################################
    ############################## PLOT MEAN ABS ERR TO NEXT NEIGHBOUR ########################################
    ###########################################################################################################

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(18,9), sharey=True)
    matplotlib.rcParams.update({'font.size': 22})
    fig.suptitle('Manhattan Distance to Next OK Image Angle ' + str(angle)  )
    ax1.imshow(img_arr_NOK[0], cmap='gray', aspect="auto")
    ax1.get_yaxis().set_visible(False)
    ax1.set_xticks([])
    ax1.set_xlabel('Example Image', labelpad= 27)
    sns.scatterplot(
        data = pd.DataFrame(
            {'MAE' :np.abs(img_arr_OK - img_arr_OK[dist_mat_OK_OK.argmin(axis = 0), :, :]).mean(axis = (0,2)), 
            'row_number' : range(252)}),
        y = 'row_number', x = 'MAE', color = 'green' )
    sns.scatterplot(
        data = pd.DataFrame(
            {'MAE' :np.abs(img_arr_NOK - img_arr_OK[dist_mat_NOK_OK.argmin(axis = 0), :, :]).mean(axis = (0,2)), 
            'row_number' : range(252)}),
            y = 'row_number', x = 'MAE', color = 'red')
    ax2.get_yaxis().set_visible(False)
    ax2.set_xlabel('Manhattan Distance per Row')
    plt.subplots_adjust(#left=.1,  bottom=0, #right=1, 
            top=.9, wspace=0, hspace=0)
    plt.legend(labels=["OK Images",'NOK Images'])
    plt.savefig('../plots/comp_MAE_pixelwise_per_row_angle' + str(angle) + '.png')
    plt.close()

    ###########################################################################################################
    ####################### PLOT DENSITY ERROR SUM TO NEXT OK NEIGHBOUR #######################################
    ###########################################################################################################    

    matplotlib.rcParams['figure.figsize'] = [15, 10]
    matplotlib.rcParams.update({'font.size': 22})
    p =sns.kdeplot(np.abs(img_arr_OK - img_arr_OK[dist_mat_OK_OK.argmin(axis = 0), :, :]  ).mean(axis = (1,2)), color = 'green')
    sns.kdeplot(np.abs(img_arr_NOK - img_arr_OK[dist_mat_NOK_OK.argmin(axis = 0), :, :]  ).mean(axis = (1,2)), color = 'red')
    p.set_xlabel('Mean Pixelwise Distance')
    plt.legend(labels=["OK Images",'NOK Images'])
    plt.title('Distribution of Distances to Nearest OK image Angle ' +  str(angle))
    plt.savefig('../plots/comp_MAE_mean_density_angle' + str(angle) + '.png')
    plt.close()


pd.DataFrame( {'OK': same_date_neighbour_OK,
'NOK': same_date_neighbour_NOK,
'All': same_date_neighbour_ALL,
'Angle': [i for i in range(1,5)]}).to_csv('../plots/same_date_neighbours.csv')