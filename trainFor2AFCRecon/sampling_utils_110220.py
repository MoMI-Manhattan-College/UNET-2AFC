#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 19:30:47 2020

@author: rachelroca and joshuaherman
A collection of functions for creating and displaying masks, as well as fourier data sampling.
"""

# Import Image object that will help with image processing
from PIL import Image

import numpy as np

# Import imshow command to see images
from matplotlib.pyplot import imshow
from data_prep_utils_110220 import normalize_pictures
from matplotlib import pyplot as plt

# Import a library for input/output in Python
import scipy.io as sio

def display_mask(masking_matrix):
    """Create figure that displays masking_matrix.

    This code takes a matrix of ones and zeros as input, and returns a figure
    containing a corresponding plot where ones show up in white and 0s in black.
    Note that in the context of masking matrices for sampling, 1s represent sampled
    frequencies and 0s represent discarded frequencies.

    Args:
        masking_matrix: a 2 dimensional numpy array of ones and zeros

    Returns:
        fig: a figure containing a grayscale image corresponding to the input matrix"""

    # This displays the image so we can see it
    fig,ax = plt.subplots(1,dpi = 1200)  # Initialize both figure and axis in only one line of code
    ax.imshow(masking_matrix,cmap = 'gray')
    plt.title('Mask')  # Set title of the image

    return fig

def sampling_mask_updated(acceleration, dim):
    """Creates a general masking matrix

    This code assigns ones and zeros to the masking matrix, where ones will be sampled.
    Low frequencies are sampled, including horizontal strips of higher frequencies, according to the acceleration chosen.

    Args:
        acceleration: A number specifying approximately how much to undersample the image. This acceleration isn't the actual acceleration but rather
        the fraction of higher frequencies to include is 1/acceleration.
        dim: The side dimension of the image to be sampled, one number for both the width and height (image is a square). 

    Returns:
         sampling_masking_matrix: masking matrix of shape dim x dim that can be utilized in
         the undersampling function
         effective_acceleration: the actual acceleration (slightly less that the acceleration variable,
         since we fully sample the low frequencies)
         """
    # Initialize mask as array of zeros
    matrix_of_ones = np.zeros((dim,dim),dtype = np.float64)

    #16 is utilized for calib_size out of recommendarion of Sajan and Angel
    calib_size = 16

    # Make sure dim and acceleration are integers
    assert(type(dim)==int)
    assert(type(acceleration)==int)

    # Sample the bands required by setting columns of matrix_of_ones equal to 1
    matrix_of_ones[0:dim:acceleration,:] = 1 # Samples the high frequencies, by first sampling across all frequencies
    matrix_of_ones[int(dim/2-calib_size/2)-1:int(dim/2+calib_size/2)-1, :] = 1 # The middle calib_size lines are sampled
    #shifted middle section onver 1 so the middle section never looks like it contains 17 lines

    # Calculate the actual acceleration
    effective_acceleration=dim**2/np.sum(matrix_of_ones)

    # Rename matrix to appropriate output name
    sampling_masking_matrix = matrix_of_ones
    return sampling_masking_matrix, effective_acceleration

def undersampling_operation(user_images,masking_matrix):
    """Performs undersampling on user_images using masking_matrix

    This code applies an undersampling scheme defined by masking_matrix to a
    set of user images in array user_images to output the corresponding set of undersampled images.

    Args:
        user_images: a 3 axis numpy array of shape m x n_H x n_W, where m is the number of images, while n_H x n_W are the dimensions
        of each individual image."
        masking_matrix: a 2 axis numpy array of shape n_H x n_W. Contains encoding for undersampling. Note that 0s in masking_matrix
        correspond to k-space frequencies to discard, while 1s correspond to k-space frequencies to keep.

    Returns:
        undersampled_images: a 3 axis numpy array of shape m x n_H x n_W. This is the array of undersampled images corresponding
        to user_images."""

    undersampled_images = np.zeros(user_images.shape[0:3])  # Initialize our Output array
    for j in range(0,user_images.shape[0]):  # Loop for each image in data set

        # Bring our image to k-space
        fourier = np.fft.fft2(user_images[j,:,:])  # Apply fourier transform
        fourier_shift = np.fft.fftshift(fourier)  # Shifting the zero and low frequency to center of image

        # Apply our masking matrix to the k-space image to get our undersampled k-space image
        accelerated_data = np.multiply(fourier_shift,masking_matrix)

        # Bring our image back to real pixel space
        inverse_fourier = np.fft.ifftshift(accelerated_data) # Reshifting the zero frequency back to corner of image
        inverse_fourier = np.fft.ifft2(inverse_fourier) # Applying the inverse fourier
        inverse_fourier = abs(inverse_fourier) #taking the magnitude for positive real values
        undersampled_images[j,:,:] = inverse_fourier  # Add undersampled image to results array
    undersampled_images = normalize_pictures(undersampled_images)

    return undersampled_images
