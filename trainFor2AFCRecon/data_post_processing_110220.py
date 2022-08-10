#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 19:54:03 2020

@author: rachelroca and joshuaherman

This file contains functions needed to provide results from the u-net reconstruction
It includes a function for plotting loss and metric values per epoch to give a training curve,
and displaying a specified reconstructed image vs its undersampled and fully sampled counterparts.

Note that this file doesn't have to be run itself, but is used as an import in the main unet file.

"""
# In[1]: Imports
import numpy as np

from matplotlib import pyplot as plt
from data_prep_utils_110220 import normalize_pictures
# %% Functions

def plot_metric(unet_architecture_train,take_root=True,training_metric_dict_key='loss',metric_display_name='Root Mean Squared Error',save_fig=False,start_file_name='',end_file_name=''):
    """Plots the Train and Validation losses vs number of epochs run

    This code creates a figure and plots the  training metric in red and the validation metric in green over
    the epochs that were used to train. This is done in the same plot. The plot is optionaly saved to disk.
    Args:
        unet_architecture_train: The object returned by running the fit method on the model for training. It includes the information on metric value vs epoch.
        take_root: This boolian variable when true takes the square root of the metric before plotting it. This is needed if we are using a mean squared error
        metric but want to display it as a root mean square metric. When false, no square root is applied.
        training_metric_dict_key: This string specified which metric we want to plot. More specifically, it is the key that is used to call the training metric
        data from the dictionary unet_architecture_train.history. Note that the key for accessing the validation data is just 'val_'+training_metric_dict_key. Thus
        specifying only the train key gives this function what it needs to know to also access the validation key without any additional specified information.
        The default value is 'loss', which will cause this function to plot the training and validation curves for the loss metric used to train this model.
        If this model is also equiped with a metric such as 'mean squared error', then 'mean squared error' can be specified as the training_metric_dict_key
        to plot the test vs training curve on this metric.
        metric_display_name: This string specifies what we want our plot to display the metric name as in the legend and y axis label. The default is
        'Root Mean Squared Error'. The labels on the legend and y axis are determined by both metric_display_name and training_metric_dict_key, in that if
        training_metric_dict_key is set to 'loss', then the labels will say 'Root Mean Squared Error Loss' and 'Root Mean Squared Error Validation Loss'.
        save_fig: Boolean that when true will result in the created figure being saved to the current directory. Otherwise the figure isn't saved to disk.
        If training_metric_dict_key is set to something other than 'loss', then the labels will say 'Root Mean Squared Error Metric' and 'Root Mean Squared Validation Metric'
        start_file_name: String that is added to the start of the saved file name if save_fig=True
        end_file_name: String that is added to the end of the saved file name if save_fig=True
    Returns:
        None
        """
    #Extracts the specified training and validation metric datas from the training history dictionary
    loss = unet_architecture_train.history[training_metric_dict_key]  # Does this for the training data metric
    val_loss = unet_architecture_train.history['val_'+training_metric_dict_key]  # Does this for the validation data metric

    # Takes the square root of the training and validation metrics only if take_root is set to True
    if take_root == True:
        loss = np.sqrt(loss)
        val_loss = np.sqrt(val_loss)

    # Creates list of indices for each epoch in loss, starting with 1 up to the number of epochs.
    epochs = range(1,len(loss)+1)

    # Will chose what kind of metric (loss or metric) will our chosen metric be referred to in our plot's labels, depending on the value of training_metric_dict_key
    if training_metric_dict_key == 'loss':
        metric_type = ' Loss'
    else:
        metric_type = ' Metric'

    # Plot the training and validation metrics on same plot vs the number of epochs so far.
    plt.figure() # Creates new blank figure on which a plot can be added
    plt.plot(epochs, loss, 'ro--', label=metric_display_name+metric_type)  # Plots the training metric in red and adds label to legend
    plt.plot(epochs, val_loss, 'go--', label=metric_display_name+' Validation'+metric_type)  # Plots the validation metric in green and adds label to legend
    # Add title, x and y labels, legend, specified scale, and grid lines
    plt.title('Training Performance')  # Our title
    plt.xlabel('Number of Epochs')  # x axis label is epochs
    plt.ylabel(metric_display_name + metric_type)  # y axis label is metric name
    plt.yscale('log') # Makes y axis that displays metric values to be in logscale
    plt.legend()  # Adds legend
    plt.grid(b=True,which='both')  # Adds grid lines to our plot
    if save_fig == True:  # Saves figure as png file to disk if True
        fname=('%sconvergence_plot_%s%s%s.png'%(start_file_name,metric_display_name,metric_type,end_file_name)).replace(" ","_").lower()  # Creates filename
        plt.savefig(fname)  # Saves as file with file name fname
    plt.show()  # Makes sure that our plot is displayed
    return None

def display_images(y_test, x_test, pred_test,image_number=9,save_fig=False,start_file_name='',end_file_name=''):
    """ Shows the original, undersampled, and reconstructed images of the same index image_number on the same figure side by side.

    This code selects the image_number = 7 image in the image test set and creates a figure
    that has all three versions of the image side by side. It also prints the RMSE between the
    reconstructed image and the original image.

    Args:
        y_test: an array of dimension, (num of test images) x dim x dim x 1, containing the fully sampled test set images
        x_test: an array of the undersampled test set images with shape:  (num of test images) x dim x dim x 1
        pred_test: an array of  (num of test images) x dim x dim x 1, of the the U-Net's reconstruction/prediction of the x_test data
        Note that dim x dim are the dimensions of an individual mri image, being dimensions height by width
        image_number: The index of the image that we want to pull from each of the three input test data sets. These three pulled images will be
        displayed side by side. The default is 7.
        save_fig: Boolean that when true will result in the created figure being saved to the current directory. Otherwise the figure isn't saved to disk
        start_file_name: String that is added to the start of the saved file name if save_fig=True
        end_file_name: String that is added to the end of the saved file name if save_fig=True
    Returns:
        None
        """

    # Create the figure and its subplots and plots the requested images to these subplots
    fig = plt.figure(figsize=(16,4.8)) # Creates a blank figure of specified size


    # Create the first subplot to the left to display the fully sampled image
    y_test = normalize_pictures(y_test)
    ax1 = fig.add_subplot(1,3,1) # Creates a blank subplot in the figure
    ax1.imshow(y_test[image_number, ...,0], cmap='gray') # Displaying the specified fully sampled image on the above created subplot
    plt.axis('off')  # Removes axis ticks from the above subplot

    # Create the second subplot in the middle to display the specified undersampled image, following same procedure as above
    x_test = normalize_pictures(x_test)
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(x_test[image_number, ...,0], cmap = "gray")
    plt.axis('off')

    # Create the third subplot to the right to display the specified reconstructed image, following the same procedure as above
    pred_test = normalize_pictures(pred_test)#normalizing
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(pred_test[image_number, ..., 0], cmap='gray')
    plt.axis('off')

    # Add the titles to each of the respective subplots
    ax1.title.set_text('Original Image')
    ax2.title.set_text('Undersampled Image')
    ax3.title.set_text('Reconstructed Image')

    if save_fig == True:  # Saves figure as png file to disk if True
        fname=('%sthree_pic_comparison_image_%s%s.png'%(start_file_name,int(image_number),end_file_name)).replace(" ","_").lower()  # Creates filename
        plt.savefig(fname)  # Saves as file with file name fname
    plt.show()
    return None
