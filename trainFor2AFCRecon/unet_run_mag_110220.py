#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 20:26:03 2020

@author: rachelroca and joshuaherman


This file trains a single unet and produces 2AFC reconstructed background and signal files containing images from a background file and signal file
that were undersampled and reconstructed using the neural network.

Things that you can change in this file:
0) Keep or remove the method that disables plots from In[0]
1a) May want to edit file directory in In[1] to specify where the dataset fully sampled mri file(s) are
1b) Pick different files for mri_images to come from if you want to use a different dataset
2) Change the parameters in In[2], which include all of the hyperparameters and undersampling
3) Change the size of the picture test set at the bottom of In[3]
4) Alter the qualifier task label in In[4], or change the unet name, or alter settings in the single_run_create_auc_pics function in In[4]

"""

# In[0] Imports
import matplotlib
matplotlib.use('Agg') #Don't display plots, for when running in screen.
from data_prep_utils_110220 import *
from data_post_processing_110220 import *

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import ModelCheckpoint
from sampling_utils_110220 import undersampling_operation,display_mask, sampling_mask_updated

from unet_unified_utils_110220 import unet_architecture as unet_architecture_u_diagram, ssim_metric, single_run_create_auc_pics


# In[1]: Load, shuffle, normalize fully sampled images

# Input file addresses for retrieving the train and validation data
directory="/home/jherman/deeplearning/dataDir"

# Retrieves the fully sampled images from the file and normalizes the images
mri_images = get_mri_images(name=directory+'/FLAIRVolumesSOS')

# In[2]:Specify model parameters, such as neural network hyperparameters and undersampling rate

loss = "MSE" # loss metric
loss_string="MSE" # name for loss metric in qualifier and by extension saved output files
epochs = 1 # Number of passes through entire training set in training
batch_size = 1 # Number of images a batch contains, where each training step only uses one batch
acceleration_skip_number = 2 # acceleration of the high frequency section of the mr kspace image (skip every acceleration_skip_number lines when sampling)
initial_filters = 4 # number of filters in convolution layer at beginning of unet
dropout = 0.1 # percent of channels to drop in dropout layer during training
dropout_string = "1dp" # how dropout level will show up in qualifier in output file names

# In[3] Get mask, undersample, normalize, and split images into train set and picture set
sampling_mask, effective_accleration = sampling_mask_updated(acceleration_skip_number,mri_images.shape[1])
print("The effective acceleration is, ", effective_accleration)
# Displays the mask as an image
display_mask(sampling_mask)
# Creates the undersampled images using this mask
undersampled_images = undersampling_operation(mri_images,sampling_mask) 

# split the image set into a train set and a small picture set. Currently set to include images 0 to 499 inclusive in the train set and the rest in the picture set
x,x_pic,y,y_pic = get_training_pic_sets(mri_images,undersampled_images,train_interval = (0,500))



# In[4]: Put together qualifier, specify unet_name, build neural network, and run single_run_create_auc_pics

#keeping track of what's being tested (metric,epochs,batch_size,accleration,initial filters,dropout,task)
qualifier = "%s_%s_%s_%s_%s_%s_SingleMagArtemis"%(loss_string,int(epochs),int(batch_size),int(acceleration_skip_number),int(initial_filters),dropout_string)


# Call unet model
input_img = Input(shape = (mri_images.shape[1],mri_images.shape[1],1))

unet_architecture=(Model(input_img, unet_architecture_u_diagram(input_img,chan=initial_filters,drop_rate=dropout)))

unet_name="Unified Diagram" # also added to output file names, Unified Diagram refers to that our
# latest version of the unet was based on a combination of two unet codes, one that Rachel made and one that I made

# Run function to train and produce 2AFC images for unet
single_run_create_auc_pics(unet_architecture,x,y,sampling_mask,unet_name,qualifier,test_signal_file_name=directory+"/FLAIR_NN_testing_signalImages_11_22_2020_Contrast5e-5",test_background_file_name = directory+"/FLAIR_NN_testing_backgroundImages_11_22_2020",x_pic = x_pic,y_pic = y_pic,pic_indices = list(range(len(x_pic))),loss=loss,optimizer = "RMSprop",batch_size=batch_size,epochs=epochs)