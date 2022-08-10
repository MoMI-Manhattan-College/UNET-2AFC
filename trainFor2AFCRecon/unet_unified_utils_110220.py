#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 19:57:12 2020

@author: rachelroca and joshuaherman

This file contains all functions needed to execute and provide figures and information of the U-Net architecture
based off of figure 6 in the FastMRI paper. You do not need to run this file, simply confirm that this file is
imported to the main file (unets_run_files) and are together in the same directory.

"""

import scipy.io

import tensorflow as tf


from tensorflow.keras.layers import Input,Conv2D,UpSampling2D, concatenate, Dropout, ReLU, MaxPooling2D
from tensorflow.image import ssim

import numpy as np
#Import imshow command to see images

#import tensorlayer as tl
from tensorflow_addons.layers import InstanceNormalization
from data_prep_utils_110220 import normalize_pictures,display_original_undersampled,get_mri_images
from data_post_processing_110220 import plot_metric,display_images
from sampling_utils_110220 import undersampling_operation




# In[1]: Define Loss functions

# Create SSIM loss function
# Based implementation from stack overflow: https://stackoverflow.com/questions/57357146/use-ssim-loss-function-with-keras
# Sukanya Dasgupta's answer is used.
def ssim_metric(y_true,y_pred):
    """Our SSIM metric.

    Defines metric based on ssim that takes true and predicted image arrays as input and outputs a single value that
    is one minus the average ssim. We chose one minus the average ssim because ssim is optimal when it is maximized, but we want a loss that is optimal
    when minimized. Since we are choosing the maximum value of the ssim to be 1, we thus subtract our ssim from 1.

    Args:
        y_true: a tensor of shape m by n_h by n_w by n_c, that contains the original image data.
        m is the number of images, while n_h by n_w are the dimensions of the image, while n_c is the number of channels.
        y_pred: a tensor of the same shape as y_true that contains the reconstructed image data that we want to compare to the original
        image data

    Returns:
        A tensor containing the value of our metric."""
    return 1-tf.reduce_mean(ssim(y_true,y_pred,1.0))



def create_auc_images(unet_architecture, unet_name, qualifier, sampling_mask, test_signal_file_name="FLAIR_NN_testing_signalImages.mat",test_background_file_name = "FLAIR_NN_testing_backgroundImages.mat"):
    """Uploads signal/background images, undersamples, reconstructs them with the neural network and exports to .mat files

    The signal images and background images are used to calculate the AUC. These images are uploaded,
    undersampled with the given mask, and then reconstructed. We then process the data to be saved in a .mat file
    where model observers can be applied.

    Args:
        unet_architecture: uploads the trained model used to reconstruct the undersampled images
        unet_name: specifies which model is being used, and is added to the file name to keep track of files better
        qualifier: specifies hyperparameter, undersampling, and other information about a file and is added to the file name
        sampling_mask: the mask used to undersample the images
        test_signal_file_name: name of mat file that contains signal images
        test_background_file_name: Name of mat file that contains background images
    Returns:
        None
       """
    # load and undersample both signal and background images
    signal_images = get_mri_images(name = test_signal_file_name, random_order = False) #getting the signal images
    signal_images = undersampling_operation(signal_images,sampling_mask) #undersampling the signal images, also gives us real data by taking the magnitude


    background_images = get_mri_images(name = test_background_file_name, random_order = False)  #getting the background images
    background_images = undersampling_operation(background_images,sampling_mask)#undersampling the background images, also gives us real data by taking the magnitude

    # reconstruct signal images and save them
    recon_signal_images = unet_architecture.predict(signal_images) #using the network to reconstruct the signal images
    #After mri_images.mat, the number of images was put as the last dimension, not first.
    #This python code is written the opposite way (shape num inages x dim x dim).
    #This switches the axes to be compatible with the rest of matlab code
    recon_signal_images = normalize_pictures(recon_signal_images) # Normalize reconstructed images
    recon_signal_images = np.moveaxis(recon_signal_images, 0, 2)
    recon_signal_images_dict = {'reconArray': recon_signal_images} #formats the images in a way it can be exported.
    scipy.io.savemat("temp_single/recon_%s_%s_%s.mat"%(test_signal_file_name.split("/")[-1],unet_name,qualifier), recon_signal_images_dict) #saves to .mat file

    # reconstruct background images and save them.
    recon_background_images = unet_architecture.predict(background_images) #using the network to reconstruct the background images
    recon_background_images = normalize_pictures(recon_background_images) # Normalize reconstructed images
    recon_background_images = np.moveaxis(recon_background_images, 0, 2) #switches axes
    recon_background_images_dict = {'reconArray': recon_background_images} #formats the images in a way it can be exported.
    scipy.io.savemat("temp_single/recon_%s_%s_%s.mat"%(test_background_file_name.split("/")[-1],unet_name,qualifier), recon_background_images_dict) #saves to .mat file

    return None

def single_run_val_auc(unet_architecture,x_train,y_train,x_val,y_val,sampling_mask,unet_name,qualifier,x_pic = None,y_pic = None,pic_indices = [],loss=ssim_metric,metric_display_name='One Minus SSIM',optimizer = "RMSprop",batch_size=16,epochs=150,test_signal_file_name="FLAIR_NN_testing_signalImages.mat",test_background_file_name = "FLAIR_NN_testing_backgroundImages.mat"):
    """Performs sinlge run training with validation on a keras neural network model for MR Image Reconstruction. AUC images created also.

    Trains a neural network using training set (x_train,y_train) and validates using validation set (x_val,y_val).
    The neural network will be trained with the specified loss function and optimizer. The resulting neural network and model summary will be saved to the folder named "temp_single", and will be scored using four metrics:
    Normalized mean squared error,Normalized Root Mean Squared Error, Normalized L1 error, and Average Structural Similarity. These values will be saved to both a txt and a mat file in the temp_single folder. Picture results and plots will also be saved there.
    AUC images will be output using signal and background data files test_signal_file_name and test_background_file_name.
    Args:
        unet_architecture: The tensorflow keras model
        x_train: The dataset containing the undersampled training images
        y_train: The dataset containing the fully sampled training images
        x_val: The dataset containing the undersampled validation images
        y_val: The dataset containing the fully sampled validation images
        x_pic: The dataset containing the undersampled picture set images
        y_pic: The dataset containing the fully sampled picture set images
        pic_indices: indices of pictures from picture set to include in picture results
            Note:The picture set is used only to produce pictures before and after neural network reconstruction
            on the selected set, to be saved to temp_single folder
        sampling_mask: The sampling mask used to undersample the images for creating x from , in this code only used in the auc image generation section

        unet_name: The name of the unet to be used in save files
        qualifier: An indicative string to be added onto the unet_name that contains further information about the model or dataset

        For example, unet_name could be "Unified Diagram" and qualifier could be "1_4_16_0dp", where the qualifier is indicative of epochs_acceleration_channels_dropout


        loss: The loss function used to train the neural network model.
        optimizer: The optimizer used to minimize the loss to train the neural network model.
        batch_size: The number of datapoints used to optimize the loss function per iteration.
        epochs: The number of times the entire dataset is run through to optimize a neural network before stopping
        the training process.
        metric_display_name: Name used to represent metric on training plots
        test_signal_file_name: The name of the mat file containing signal images
        test_background_file_name: The name of the mat file containing background images
    Returns:
        scores: dictionary containing metric scores on validation set"""
        
    # display selected images full and undersampled side by side
    for i in pic_indices:
        display_original_undersampled(y_pic, x_pic, image_number = i,save_fig=True,start_file_name='temp_single/',end_file_name='_%s_%s'%(unet_name,qualifier))





    # Reshape all data by adding channel dimension
    x_pic = x_pic.reshape((*x_pic.shape,1))
    y_pic = y_pic.reshape((*y_pic.shape,1))

    x_train=x_train.reshape((*x_train.shape,1))
    x_val=x_val.reshape((*x_val.shape,1))
    y_train=y_train.reshape((*y_train.shape,1))
    y_val=y_val.reshape((*y_val.shape,1))

    # group train and val data to be used later
    train = (x_train,y_train)
    val = (x_val,y_val)

    # change the name of the model variable (only to make adapting this code from cross val easier)
    model=unet_architecture

    #Save the summary for the current neural network model
    file=open('temp_single/summary_%s.txt'%((unet_name + qualifier).replace(" ","_").lower()),'w')
    print_fn=lambda x: print(x,file=file)
    model.summary(print_fn=print_fn) # Displays a nice breakdown of our model
    file.close()

    # Compile the given model for the given loss and optimizer
    model.compile(loss=loss,optimizer=optimizer)

    # Fit the given model
    model_train = model.fit(*train,validation_data=val,batch_size=batch_size,epochs=epochs)


    # Save the model, plot and save convergence plot
    model.save('temp_single/model_%s.h5'%((unet_name + qualifier).replace(" ","_").lower()))
    plot_metric(model_train, take_root=False,training_metric_dict_key='loss',metric_display_name=metric_display_name,save_fig=True,start_file_name='temp_single/%s_'%((unet_name + qualifier).replace(" ","_").lower()))
    
    # reconstruct undersampled picture set and add to array that will later be saved to file, and save images of original side by side with undersampled and reconstructed images from the picture set
    pred_pic=model.predict(x_pic)  # Neural network reconstruction
    pred_pic=normalize_pictures(pred_pic)  # Normalize the results
    pred_pics_array=pred_pic[:,...,0]
    for i in pic_indices:
        display_images(y_pic, x_pic, pred_pic,image_number=i,save_fig=True,start_file_name='temp_single/',end_file_name='_%s_%s'%(unet_name,qualifier))


    # Reconstruct the validation set pictures from the undersampled picture array x_val
    y_pred=model.predict(x_val)  # Neural network reconstruction
    y_pred=normalize_pictures(y_pred)  # Normalize the results


    # convert validation and reconstruction data to tensor form of same precision for scoring 
    y_pred = tf.convert_to_tensor(y_pred,dtype=tf.float64)
    y_val = tf.convert_to_tensor(y_val,dtype=tf.float64)

    # Score the model using the three metrics
    # Calculating SSIM metric
    average_ssim = tf.reduce_mean(ssim(y_val,y_pred,1.0))
    # Calculating NMSE, matching with equation 7 in fastMRI paper
    nmse = (tf.norm((y_pred - y_val), ord = 'euclidean'))**2/(tf.norm(y_val, ord = 'euclidean'))**2
    # Calculating NRMSE
    nrmse = tf.norm((y_pred - y_val), ord = 'euclidean')/tf.norm(y_val, ord = 'euclidean')
    # Calculating L1
    norml1 = tf.norm((y_pred - y_val), ord = 1)/(tf.norm(y_val, ord = 1))

    # Convert tensor scores to numpy
    average_ssim=average_ssim.numpy()
    nmse=nmse.numpy()
    nrmse=nrmse.numpy()
    norml1=norml1.numpy()



    # Save the metric scores.
    # Opening or creating a file if it doesn't already exist
    f = open("temp_single/average_loss_calc_"+unet_name + qualifier + ".txt", "w")
    # Writing the metrics to the file

    f.write("ssim = " + str(average_ssim) + '\n')
    f.write("nmse = " + str(nmse) +'\n')
    f.write("nrmse = " + str(nrmse) +'\n')
    f.write("norml1 = " + str(norml1) + '\n')

    # closing the file
    f.close()

    # Also save scores and save test pictures in dictionaries to mat file
    scores={
     "ssim":average_ssim,
     "nmse":nmse,
     "nrmse":nrmse,
     "norml1":norml1
     }

    test_pictures={"x_pic":x_pic[:,...,0],
     "y_pic":y_pic[:,...,0],
     "pred_pics":pred_pics_array}

    scipy.io.savemat("temp_single/average_loss_calc_"+unet_name + "_"+qualifier+".mat",scores)
    scipy.io.savemat("temp_single/test_pictures_" + unet_name +"_"+ qualifier +".mat",test_pictures)
    
    # Create reconstructions of undersampled signal/background pictures for auc and save to mat file
    create_auc_images(model, unet_name, qualifier, sampling_mask, test_signal_file_name,test_background_file_name)
    return scores


def single_run_create_auc_pics(unet_architecture,x,y,sampling_mask,unet_name,qualifier,test_signal_file_name="FLAIR_NN_testing_signalImages.mat",test_background_file_name = "FLAIR_NN_testing_backgroundImages.mat",x_pic = None,y_pic = None,pic_indices = [],loss=ssim_metric,optimizer = "RMSprop",batch_size=16,epochs=150):
    """Performs training of a keras neural network model for MR Image Reconstruction and outputs reconstructions of given AUC signal and background data.

    Trains a neural network using training set (x_train,y_train) without validating.
    The neural network will be trained with the specified loss function and optimizer. The resulting neural network and model summary will be saved to the folder named "temp_single", and will be scored using four metrics:
    Normalized mean squared error,Normalized Root Mean Squared Error, Normalized L1 error, and Average Structural Similarity. These values will be saved to both a txt and a mat file in the temp_single folder. Picture results will also be saved there.
    AUC images will be output using signal and background data files test_signal_file_name and test_background_file_name.
   
    Args:
        unet_architecture: The tensorflow keras model
        x: The dataset containing the undersampled images
        y: The dataset containing the fully sampled images
        sampling_mask: sampling mask to be used for creating AUC images
        unet_name: The name of the unet to be used in save files
        qualifier: An indicative string to be added onto the unet_name that contains further information about the model or dataset

        For example, unet_name could be "Unified Diagram" and qualifier could be "1_4_16_0dp", where the qualifier is indicative of epochs_acceleration_channels_dropout

        test_signal_file_name: file name of signal images
        test_background_file_name: file name of background images
        x_pic,y_pic: undersampled and original picture dataset respectively
        pic_indices: Indices of pictures from picture dataset to save as images
        
        loss: The loss function used to train the neural network model.
        optimizer: The optimizer used to minimize the loss to train the neural network model.
        batch_size: The number of datapoints used to optimzize the loss function per iteration.
        epochs: The number of times the entire dataset is run through to optimize a neural network before stopping
        the training process.

    Returns:
        None"""
        
    # produce images of fully sampled and undersampled images side by side from the picture dataset
    for i in pic_indices:
        display_original_undersampled(y_pic, x_pic, image_number = i,save_fig=True,start_file_name='temp_single/',end_file_name='_%s_%s'%(unet_name,qualifier))


    # Add channel dimension at end of different data arrays for compatibility with tensorflow

    x= x.reshape((*x.shape,1))
    y= y.reshape((*y.shape,1))


    x_pic = x_pic.reshape((*x_pic.shape,1))
    y_pic = y_pic.reshape((*y_pic.shape,1))


    # group training data together in tupple for later use
    train = (x,y)





    # changing name to model since model is used in rest of code
    model=unet_architecture

    #Save the summary for the current neural network model
    file=open('temp_single/summary_%s.txt'%((unet_name + qualifier).replace(" ","_").lower()),'w')
    print_fn=lambda x: print(x,file=file)
    model.summary(print_fn=print_fn) # Displays a nice breakdown of our model
    file.close()

    # Compile the given model for the given loss and optimizer
    model.compile(loss=loss,optimizer=optimizer)

    # Fit the given model to the given validation-train split
    model_train = model.fit(*train,batch_size=batch_size,epochs=epochs)


    # Save the model
    model.save('temp_single/model_%s.h5'%((unet_name + qualifier).replace(" ","_").lower()))

    #Reconstruct undersampled picture images
    pred_pic=model.predict(x_pic)  # Neural network reconstruction
    pred_pic=normalize_pictures(pred_pic)  # Normalize the results

    # display/save selected parts of picture set, showing original by undersampled by reconstructed in each image
    for i in pic_indices:
        display_images(y_pic, x_pic, pred_pic,image_number=i,save_fig=True,start_file_name='temp_single/',end_file_name='_%s_%s'%(unet_name,qualifier))

    # Put test_pictures into dictionary and save as mat file
    test_pictures={"x_pic":x_pic[:,...,0],
     "y_pic":y_pic[:,...,0],
     "pred_pic":pred_pic}

    scipy.io.savemat("temp_single/test_pictures_" + unet_name +"_"+ qualifier +".mat",test_pictures)

    # Save reconstructions of 2AFC signal and background images to mat files for 2AFC and AUC use
    create_auc_images(model, unet_name, qualifier, sampling_mask, test_signal_file_name,test_background_file_name)



# %% Define U-Net blocks and then U-Net model
# Below we define our unet blocks as functions
def conv_block(input_layer,filters,drop_rate):
    """This block is used in our U-net to apply all of our convolutions in the expanding and contracting paths.

    This block applies a convolution followed by ReLU, instance normalization, and dropout.
    It serves as a building block for the conv_block_down and conv_block_up functions.

    Args:
        filters: The number of filters applied in the convolution layer
        drop_rate: The percentage of channels dropped for each image input into the dropout layers
        input_layer: The layer sequence that conv_block functionally acts on

    Returns:
        output_layer: The resulting layer sequence composed of conv_block acting on input_layer"""

    output_layer = Conv2D(filters, (3, 3), strides = (1,1), padding='same')(input_layer)
    output_layer = ReLU()(output_layer)
    output_layer = InstanceNormalization()(output_layer)
    output_layer = Dropout(rate=drop_rate)(output_layer)

    return output_layer

def conv_block_down(input_layer,filters,drop_rate=0.1):
    """This block is used within our U-net during the contracting path.

    This block applies the conv_block function twice, followed by a Max pooling layer.

    Args:
        filters: The number of filters applied in each of the two convolution layers
        drop_rate: The percentage of channels to drop in the dropout layers
        input_layer: The layer sequence that conv_block_up functionaly acts on

    Returns:
        output_block: The resulting layer sequence composed of conv_block_down acting on input_layer
        conv_down: The layer we use to merge during the expanding path"""

    conv_down = conv_block(input_layer,filters,drop_rate=drop_rate)
    conv_down = conv_block(conv_down,filters,drop_rate=drop_rate)
    output_block = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_down)

    return output_block,conv_down

def conv_block_up(input_layer,filters,drop_rate=0.1):
    """This block is used in our U-net during the expanding path.

    This block applies the conv_block function twice, followed by a bilinear Upsampling layer.

    Args:
        filters: The number of filters applied in both of the convolution layers,
        drop_rate: The percentage of channels to drop in the dropout layers.
        input_layer: The layer sequence that conv_block_up functionaly acts on.

    Returns:
        output_block: The resulting layer sequence composed of conv_block_up acting on input_layer"""

    conv_up = conv_block(input_layer,filters,drop_rate=drop_rate)
    conv_up = conv_block(conv_up,filters,drop_rate=drop_rate)
    output_block = UpSampling2D((2,2), interpolation = 'bilinear')(conv_up)

    return output_block


def conv(input_layer,filters,drop_rate=0.1):
    """This block is used in our U-net at the end of the 'U'.

    This block applies the conv_block function once.

    Args:
        filters: The number of filters applied in the convolution layer,
        drop_rate: The percentage of channels to drop in the dropout layers.
        input_layer: The layer sequence that conv functionaly acts on.

    Returns:
        output_layer: The resulting layer sequence composed of conv acting on input_layer"""
    output_block = conv_block(input_layer,filters,drop_rate=drop_rate)
    return output_block

# Below is our unet defined as a function
def unet_architecture(input_img = Input(shape = (320,320,1)),chan=64,drop_rate=0.1):
    """A unet structure based on the fastMRI implementation.

Add this to the unified unet documentation as it is edited.

    The U-Net structure consists of two paths and their interconnections. The two paths are a contracting path followed by and expanding path.
    We have a model that starts with a specified number of filters, chan (or x in the diagram), in its first convolutional block and has 4 pooling layers in the contracting path and 4 upsampling layers in the expanding path.

    We start this model with two convolutional blocks in sequence. The convolutional block is the function conv_block.
    After this pair of convolutional blocks, we apply a max pooling layer that halves the number of features in both
    height and width dimensions. This max pooling layer is followed by another two convolutional block sequence that has double the number of filters, followed by another max pooling layer. This is repeated two more times until a total of 4 max pooling layers are used.
     We then have a single convolutional block after this fourth pooling layer that has the same number of filters as the previous two convolutional blocks. This marks the bottom of the path.
    This convolution block has the most filters, which is 8x. The size of the image, which at the beginning of the network is 320 x 320 x 1, is now 20 x 20 x 8x, where 8x is the number of channels.
    Then the expansive path is started with a bilinear upsampling layer, which doubles the number of features in both width and height dimensions. It is used within the conv_block_up function to move up the U.
    Before each upsampling, we merge the data on the opposite side of the U pre-pooling to the data at the current layer.
    For example, the first max-pooling layer is opposite in the "U" from the fourth upsampling layer, just as the
    second is from the third, and the third is from the second, and the fourth max-pooling layer is opposite from the first upsampling layer.
    Then the concatenation is fed into a sequence of two new convolutional blocks each having half the number of filters as the previous convolutional block.

    This process of upsampling folowed by concatenation followed by two convolutional blocks is repeated until four concatenations are made.
    Note that the two convolutional blocks after the last concatenation will have the same number of filters as the previous sequence of convolutional blocks in contrast to the preceding part of this pattern.
    This last convolutional block has x filters, which we started off with.  The output of this
    convolution block is then fed into 2 convolution layers that use 1 by 1 filters as well as only chan/2 and 1 filters respectively,
    and a relu and sigmoid activation function respectively. Same padding is used in these two convolutional layers. This marks the end of the expansive path, and the resulting image has the same dimensions as the beginning image, which is 320 x 320 x 1.


    Args:
        input_img: Layer that this network architecture acts on functionaly to yield our full network. It defaults to an input layer
        having the shape of a single 64 by 64 by 1 image.
        chan: the number of filters that our first convolutional block's convolutional layers use. It defaults to 64 filters, but is easily changed to other values.
        drop_rate: The percentage of channels that are dropped in the dropout layers embedded within our convolutional blocks. It defaults
        to 0.1, which is 10 percent drop out
    Returns:
        conv_end_block: Layer that is output by this network composed on input_img."""

    conv_block_down_one,merge1 = conv_block_down(input_img,chan,drop_rate)  # First convolution block with chan filters
    chan*=2
    conv_block_down_two,merge2 = conv_block_down(conv_block_down_one,chan,drop_rate)
    chan*=2
    conv_block_down_three,merge3 = conv_block_down(conv_block_down_two,chan,drop_rate)
    chan*=2
    conv_block_down_four,merge4 = conv_block_down(conv_block_down_three,chan,drop_rate)
    conv_block_down_five = conv(conv_block_down_four,chan,drop_rate)
    # Reached the bottom of the Unet Architecture.
    chan//=2
    conv_block_up_six = UpSampling2D((2,2), interpolation = 'bilinear')(conv_block_down_five)
    conv_block_up_six = concatenate([merge4,conv_block_up_six],axis=-1)
    conv_block_up_six = conv_block_up(conv_block_up_six,chan,drop_rate)
    merge5 = concatenate([merge3, conv_block_up_six], axis = -1)
    chan//=2
    conv_block_up_seven = conv_block_up(merge5, chan,drop_rate)
    merge6= concatenate([merge2,conv_block_up_seven], axis = -1)
    chan//=2
    conv_block_up_seven = conv_block_up(merge6, chan,drop_rate)
    merge7 = concatenate([merge1,conv_block_up_seven])
    conv_end_block = conv(merge7,chan,drop_rate) #we don't go up any any more blocks in the archiecture so use conv function instead of conv_block_up
    conv_end_block = conv(conv_end_block,chan,drop_rate)
    chan//=2
    conv_end_block = Conv2D(chan, 1, activation='relu', padding='same')(conv_end_block)
    conv_end_block = Conv2D(1, 1, activation='sigmoid', padding='same')(conv_end_block)  # One by One convolution as end layer

    return conv_end_block
