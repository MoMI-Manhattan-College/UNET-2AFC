#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 20:26:03 2020

@author: Rachel Roca and Joshua Herman

This is the main file for training the unet on the image dataset and using the trained unet to produce 2AFC files for Human Observer Studies and Model Observer AUC scores

The code is run from this file.

Things that you can change in this file:
0) Keep or remove the method that disables plots from In[0] (Imports)
1) May want to edit filename in get_mri_images function in In[1] (Load, shuffle ...) lines 39 & 40, if mri image file stored in different folder
2) Change the parameters in In[1](Load, shuffle ...), which include all of the hyperparameters and undersampling. Some alternative parameters are commented out.
    Lines 43 and 45 have commented out parameters for SSIM loss, while 44 and 46 have parameters for MSE loss. commented out lines 49 and 54 are suggested settings for running a quick test that the code works as intended.
3a) Change qualifier label in In[2] (Qualifier is put together...) from 2AFC to something more descriptive if want to
3b) Change unet_name in In[2] (Qualifier is put together...) if want to alter filenames of files output from unet training/2AFC file generation function
4) Alter parameters of unet training/2AFC file generation function in In[2](Qualifier is put together...), such as trying Adam instead of RMSProp
5) Change the size of the training set in get_training_pic_sets at the bottom of In[1]
"""
# In[0] Imports
import matplotlib
#matplotlib.use('Agg') #Don't display plots, for when running in screen.

#tensorflow
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# utilities
from data_prep_utils import *
from data_post_processing_utils import *
from sampling_utils import undersampling_operation,display_mask, get_sampling_mask
from unet_utils import unet_architecture as get_unet_architecture, ssim_metric, single_run_create_2afc_pics

# In[1]: Load, shuffle, undersample, normalize, and plot picture data. Set parameters for unet.

# Retrieves the fully sampled images from the file and normalizes the images
directory="../"
mri_images = get_mri_images(name=directory+'/training_sample.mat')

# Specify parameters for network and for qualifier (important in output file names) and for convergence plot labels
#loss = ssim_metric  # The loss function
loss = "MSE"
#loss_string="SSIM"  # Loss function name for qualifier
loss_string="MSE"

epochs = 150 # Number of epochs to train
#epochs=1 # for quick test to see that code runs
batch_size = 8 # Batch size for training
acceleration_skip_number = 3 # Acceleration, specificaly is k where we skip every k lines of high frequency in kx undersampling
# =2, 3, 4, or 5
initial_filters = 64 # initial filters in unet
#initial_filters = 2 # for quick test to see that code runs
dropout = 0.1 # dropout rate in u-net
dropout_string = "1dp" # how dropout rate will be displayed in qualifier

# Creates the k-space sampling mask we will be using
sampling_mask, effective_accleration = get_sampling_mask(acceleration_skip_number,mri_images.shape[1])
print("The effective acceleration is, ", effective_accleration)
# Displays the mask as an image
display_mask(sampling_mask)
# Creates the undersampled images using this mask
undersampled_images = undersampling_operation(mri_images,sampling_mask)
x,x_pic,y,y_pic = get_training_pic_sets(mri_images,undersampled_images,train_interval = (0,100)) # first 100 images for training, the rest for picture test set


# In[2]: Qualifier is put together, unet_name is specified, unet model is created, and unet model is trained to create 2AFC images

#keeping track of what's being tested (metric,epochs,batch_size,accleration,initial filters,dropout,task)
qualifier = "%s_%s_%s_%s_%s_%s_2AFC"%(loss_string,int(epochs),int(batch_size),int(acceleration_skip_number),int(initial_filters),dropout_string)

# Call unet model
input_img = Input(shape = (mri_images.shape[1],mri_images.shape[1],1))

unet_architecture=(Model(input_img, get_unet_architecture(input_img,chan=initial_filters,drop_rate=dropout)))

unet_name="unet" # Useful in file output names

# Run function to train unet and produce 2AFC images for unet
single_run_create_2afc_pics(unet_architecture,x,y,sampling_mask,unet_name,qualifier,test_signal_file_name=directory+"/signalImages_sample",test_background_file_name = directory+"/backgroundImages_sample",x_pic = x_pic,y_pic = y_pic,pic_indices = list(range(len(x_pic))),loss=loss,optimizer = "RMSprop",batch_size=batch_size,epochs=epochs)