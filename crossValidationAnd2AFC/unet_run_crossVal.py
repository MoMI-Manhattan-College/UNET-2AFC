#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 20:26:03 2020

@author: Rachel Roca and Joshua Herman

This is the main file for running cross validation on the image dataset

The code is run from this file.

Things that you can change in this file:
0) Keep or remove the method that disables plots from In[0] (Imports)
1) May want to edit filename in get_mri_images function in In[1] (Load, shuffle ...) lines 39 & 40, if mri image file stored in different folder
2) Change the parameters in In[1](Load, shuffle ...), which include all of the hyperparameters and undersampling. Some alternative parameters are commented out.
    Lines 43, 45, and 47 have parameters for SSIM loss, while 44, 46, and 48 have have commented out parameters for MSE loss. commented out lines 51 and 56 are suggested settings for running a quick test that the code works as intended.
3a) Change qualifier label in In[2] (Qualifier is put together...) from CVMAG to something more descriptive if want to
3b) Change unet_name in In[2] (Qualifier is put together...) if want to alter filenames of files output from cross validation function
4) Alter parameters of kfold validation function in In[2](Qualifier is put together...), such as trying Adam instead of RMSProp
5) Change the size of the training set in get_training_pic_sets at the bottom of In[1]
"""
# In[0]: Imports
import matplotlib
#matplotlib.use('Agg') #Don't display plots, for when running in screen.

#tensorflow
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# utilities
from data_prep_utils import *
from data_post_processing_utils import *
from sampling_utils import undersampling_operation,display_mask, get_sampling_mask
from unet_utils import unet_architecture as get_unet_architecture, ssim_metric, k_fold_validation

# In[1]: Load, shuffle, undersample, normalize, and plot picture data. Set parameters for unet.

# Retrieves the fully sampled images from the file and normalizes the images
directory="../"
mri_images = get_mri_images(name=directory+'/training_sample')

# Specify parameters for network and for qualifier (important in output file names) and for convergence plot labels
loss = ssim_metric  # The loss function
#loss = "MSE"
loss_string="SSIM"  # Loss function name for qualifier
#loss_string="MSE"
metric_display_name='One Minus SSIM'  # Loss function name for convergence plot
#metric_display_name='MSE'

#epochs = 150 # Number of epochs to train
epochs=1 # for quick test to see that code runs
batch_size = 8 # Batch size for training
acceleration_skip_number = 3 # Acceleration, specificaly is k where we skip every k lines of high frequency in kx undersampling
# =2, 3, 4, or 5
#initial_filters = 64 # initial filters in unet
initial_filters = 2 # for quick test to see that code runs
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



# In[2]: Qualifier is put together, unet_name is specified, unet model is created, and k-fold cross-validation is executed

#keeping track of what's being tested (metric,epochs,batch_size,accleration,initial filters,dropout,task)
qualifier = "%s_%s_%s_%s_%s_%s_CVMag"%(loss_string,int(epochs),int(batch_size),int(acceleration_skip_number),int(initial_filters),dropout_string)


# Call unet model
input_img = Input(shape = (mri_images.shape[1],mri_images.shape[1],1))

unet_architecture=(Model(input_img, get_unet_architecture(input_img,chan=initial_filters,drop_rate=dropout)))

unet_name="Unet" # Useful in file output names


# k-fold validation
k_fold_validation(unet_architecture,x,y,unet_name,qualifier,x_pic = x_pic, y_pic = y_pic, pic_indices = range(len(x_pic)), num_splits=5,loss=loss,optimizer="RMSprop",metric_display_name=metric_display_name,batch_size=batch_size,epochs=epochs)
