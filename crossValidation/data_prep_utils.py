#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 19:33:17 2020

@author: Rachel Roca and Joshua Herman


This file contains functions for preparing the mri data prior to feeding into a reconstruction algorithm.
This includes a function for loading, shuffling, and normalizing the data as well as a function for just normalizing the pictures.
Also included is a function for displaying a fully sampled image beside its undersampled counterpart, and a function for splitting
the data into a training set and a set to just use as picture results.
You do not need to run this file, simply confirm that this file is imported to the main file (unet_run)
and are together in the same directory.

"""

import numpy as np
from matplotlib import pyplot as plt
import  scipy.io as sio
from random import shuffle, seed

def normalize_pictures(mri_images,dtype=None):
    """Normalizes each image from an image array such that each image has a minimum pixel value of 0 and a maximum pixel value of 1.
    Args:
        mri_images: The numpy array containing image space mri images. The array is
        of shape m x n_h x n_w x 1, where m is the number of images, n_h x n_w are the dimensions of an individual image,
        and 1 is the number of channels. Note that the channel dimension is optional for this code to work.
        dtype: Datatype of outputs
    Returns:
        normalized_mri_images: The numpy array of same shape as mri_images that contains their normalized counterparts."""

    # if data type set as None, set datatype of outputs to be same as inputs
    if dtype == None:
        dtype = type(mri_images.reshape(-1)[0])
    normalized_mri_images=np.zeros(mri_images.shape,dtype=dtype)

    for n in range(len(mri_images)):  # Normalize each image one at a time
        normalized_mri_images[n] = (mri_images[n]-mri_images[n].min())  # Make minimum pixel be zero
        normalized_mri_images[n] = normalized_mri_images[n]/normalized_mri_images[n].max() # Maxe maximum pixel be one
    return normalized_mri_images

#%%
def display_original_undersampled(mri_images, undersampled_images, image_number = 9,save_fig=False,start_file_name='',end_file_name=''):
    """Displays a given original image from the data set, and the respective undersampled image.

    This function takes the set of fully sampled images in array mri_images and undersampled images in array undersampled_images,
    as well as the index image_number of the image that we want to display from both datasets. The function then displays the two images
    from the two respective datasets that have this index side by side.


    Args:
        mri_images: The numpy array containing original fully sampled image space mri images. The array is
        of shape m x n_h x n_w, where m is the number of images, n_h x n_w are the dimensions of an individual image.
        undersampled_images: An array of the same shape as mri_images, that contains the corresponding undersampled mri images.
        image_number: the index of the image that we wish to call from both arrays to display side by side.
        save_fig: The boolean variable that when True will have the figure saved to the current directory.
        start_file_name: The string added to the beginning of the default file save name if save_fig=True.
        end_file_name: The string added to the end of the default file save name if save_fig=True
        Note that the default save name is "original_vs_undersampled_image_" + image_number + ".png"
    Returns:
         None"""
    blurred_mri_image = undersampled_images[image_number] # Extracts our chosen image from the set of undersampled images

    original_image_mri = mri_images[image_number]# Extracts our chosen fully sampled image

    # creates the figure, plots the images, adds titles
    fig = plt.figure(figsize=(10,4.8)) # Creates blank figure
    ax1 = fig.add_subplot(121)  # adds a blank plot on left
    plt.axis('off')
    ax2 = fig.add_subplot(122)  # adds a blank plot on right
    plt.axis('off')

    # Adds the two images to these two respective blank plots
    ax1.imshow(original_image_mri, cmap = "gray") # chosen cmap argument makes colormap grayscale

    ax2.imshow(blurred_mri_image, cmap = "gray")


    # Adds titles to the two plots
    ax1.title.set_text("Original Image")
    ax2.title.set_text("Undersampled Image")

    if save_fig == True:  # Saves figure as png file to disk if True
        fname=('%soriginal_vs_undersampled_image_%s%s.png'%(start_file_name,int(image_number),end_file_name)).replace(" ","_").lower()  # Creates filename
        plt.savefig(fname)  # Saves as file with file name fname
    plt.show()
    return None

#%%

def get_mri_images(shuffle_seed=78, name = 'training_sample.mat', random_order = True):
    """Retrieves the MRI images

    This code retrieves the MRI image array that contains training and validation sets from 'training.mat' by default. This set contains complex number data.
    Then it shuffles the image indexing order, reorganizes the axes of the image, takes the absolute value of the images to make them real,
    and normalizes them so each image has a minimum pixel value of 0 and maximum of 1.
    This code returns the resulting images in an array. In our default file choice, there are 110 FLAIR (Fluid Attenuation Inversion Recovery) slices of shape 320 x 320.
    The number of images have varied in our research for our corresponding publications on the use of the U-Net for Accelerated MR Imaging Reconstruction but not the dimensions 320 x 320.
    

    Args:
        shuffle_seed: This value fixes the random behavior of the function's data shuffling, so the same order results every time
        the same seed value is used.
        name: a string containing the name of the file we are uploading image from. The default is 'training.mat',
        where we get our training data from.
        random_order: boolean that determines whether the order of images will be randomly shuffled (True) or not (False)

    Returns:
        mri_images: An array containing the shuffled and normalized mri images. This array is of shape (number of images) x nh x nw
        where nh x nw is the shape of each image. For our default file choice, the output shape is 510 x 320 x 320"""

    # Load the matlab file as a dictionary
    # Note: This line of code requires that matlab file is in the Python working directory
    mri_images_dictionary = sio.loadmat(name)
    # we extract the MRI data array from the dictionary
    mri_images = mri_images_dictionary['reconArray'] #for reference, if you don't know the key,
    # check the mri_images_dictionary value (click on it) in the variable explorers. It lists all keys and values.
    # Next randomly shuffle the images
    if random_order:
        seed(shuffle_seed) # Fixes random activity to specified preset
        ind_list = [i for i in range(mri_images.shape[-1])] # list of ordered indices 0 to 510
        shuffle(ind_list)  # shuffles the indices
        mri_images  = mri_images[:,:, ind_list]  # Uses shuffled indices to shuffle the images

    #After mri_images.mat, the number of images was put as the last dimension, not first. This code switches the axes
    #to be compatible with the rest of our code (shape num inages x dim x dim)
    mri_images = np.moveaxis(mri_images, 2, 0)
    mri_images = abs(mri_images)
    mri_images = normalize_pictures(mri_images) # Now normalize each image to have pixel values min of 0 and max of 1

    return mri_images

def get_training_pic_sets(mri_images,undersampled_images,train_interval = (0,100)):
    """Splits dataset into two parts.
    
    This function splits the two arrays of mri_images and undersampled images into a training set and picture set component.
    The picture set is meant to be used as a test set just for image results, while the train set is used to train the neural network,
    although the resulting split could be used for any purpose.
    
    Args:
    mri_images: Array of fully sampled images where first array dimension is number of images
    undersampled_images: Array of undersampled images where first array dimension is number of images
    train_interval: Tuple that specifies the range of indices of images to include in the training set.
    Note that the first index in the tupple is included while the last is not. The rest of the images not in this range
    are placed in the picture set.
    
    Returns:
    x_train: The training set of undersampled images, taken from undersampled_images
    x_pic: The picture set of undersampled images, taken from undersampled images
    y_train: The training set of fully sampled images, taken from mri_images
    y_pic: The picture set of fully sampled images, taken from mri_images"""

    x_train = undersampled_images[train_interval[0]:train_interval[1]] # Takes the first specified images and assigns it to x train set
    x_pic =  undersampled_images[train_interval[1]:] # Takes the next specified images and puts it into x picture set


    y_train = mri_images[train_interval[0]:train_interval[1]] # Takes the first specified images and assigns it to y train set
    y_pic = mri_images[train_interval[1]:] # Takes the next specified images and puts it into y picture set
    return x_train,x_pic,y_train,y_pic
