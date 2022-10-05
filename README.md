# UNET Neural Network for Undersampled MRI and 2-AFC Studies

This repository contains code used in a submission to Magnetic Resonance Imaging:

JD Herman, RE Roca, AG O'Neill, ML Wong, SG Lingala, and AR Pineda, "Task-based Assessment for Neural Networks: Evaluating Undersampled MRI Reconstructions 
based on Human Observer Signal Detection:, Magnetic Resonance Imaging (under review) 

The purpose of this folder and its contents is to train neural networks to simulate undersampled MRI for the purpose of testing a new metric based on 
a two alternative-forced choice task (2-AFC) where a person decides which of two images contain a subtle signal.  We compute both standard metrics of image quality 
NRMSE (normalized root mean squared error) and SSIM (structural similarity) in a cross-validation study and generate images for a 2-AFC study.

This folder contains three data files, one yaml file called UNET.yml, and one subfolder called crossValidationAnd2AFC. The data files are "training_sample.mat",
"backgroundImages_sample.mat", and "signalImages_sample.mat." The first of these files is used for training the neural network, while the last two is used for 
creating undersampled background and signal images from a trained U-Net.

The data files should be kept in the same folder as the subfolder, crossValidationAnd2AFC.  The subfolder contains all neural network related utility 
files and runnable scripts to run as well. It also contains an "outputs" folder as a folder to collect outputs produced by the run files. 
These are described by the README file stored within the subfolder.

Note that the code in the subfolder runs using modules including TensorFlow and TensorFlow Addons.
To set up the anaconda environment to use this code, use the command 
"conda env create -f UNET.yml" to create it using the settings from the yaml file.
To activate this environment you can use "conda activate UNET".
"conda deactivate" can be used to return to the base environment.
