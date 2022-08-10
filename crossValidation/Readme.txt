August 2021
By Joshua Herman
Code Written by Joshua Herman and Rachel Roca

This folder is "crossValidation". It contains the cross validation code for the unet. Each of these files contain documentation
such as docstrings at the top of the file that gives an idea of what the file is about.

This folder contains one file which can be immediately run, while the rest are utilities containing supporting functions.
This runnable file is for running the cross validation and is as follows:

1) unet_run.py is the file set up to run cross validation on the small dataset '../training_sample'. You may want to change certain parameters 
in the file such as the directory, file name of the fully sampled mri images, the hyperparameters of the neural network and undersampling, and display information such as the name of the loss function on convergence 
plots or the qualifier strip that ends up in output filenames. Most importantly are the hyperparameters, kx acceleration level, and qualifier components that are shown in In[1].
You must change these to specify what kind of network you want to run and how you want the files output from the code to display the associated information in their name. This file is practicaly identical to the code
used in the paper. Note the docstring at the top of the file that mentions the kinds of things that you might want to change in the file between runs.


This folder also has 4 utility files that aren't runnable but must be in the same folder as the runnable file.

2) data_post_processing_utils.py contains the functions for plotting the convergence plots and the code for displaying the fully sampled image besides the undersampled image beside the reconstructed image.

3) data_prep_utils.py contains the functions for normalizing the images, displaying a fully sampled image besides its undersampled counterpart, and loading/shuffling/normalizing the fully sampled mri images from file, and a function for splitting the data into a training set and a set for test images.

4) sampling_utils.py contains the functions for creating and displaying a k-space sampling mask as well as applying it to undersample data.

5) unet_utils.py contains functions for all things neural network and cross validation related, such as the ssim metric, the k fold validation function, the unet and its block components


Outputs of file 1) which runs cross validation on the U-Net: All outputs are sent to the "outputs" folder. If you don't have this folder in the directory of unet_run.py, be sure to create it.
a) average_loss_calc_Unet_MSE_150_8_2_64_1dp_CVMag.mat stores the ssim (structural similarity), nmse (normalized mean squared error), nrmse (normalized root mean squared error) and norml1 (normalized l1 norm error) scores, means, and standard deviations for a U-Net with MSE loss, 150 epochs, batch size of 8,
undersampling factor k=2 (kx sampling), 64 initial filters, and dropout rate of 0.1 (1dp), and the label CVMag for example indicates that we were using cross validation on a magnitude image training set.
"Unet" is indicative that we are calling our U-Net model "Unet" in the file save names. 
Note that a) through h) involve models with the same parameters.

b) average_loss_calc_UnetMSE_150_8_2_64_1dp_CVMag.txt is the same as the mat file but everything is in ".txt" format
c) model_unetmse_150_8_2_64_1dp_cvmag_split1.h5 is a saved trained neural network model run for the first split of 5 fold cross validation.
d) original_vs_undersampled_image_0_unet_mse_150_8_2_64_1dp_cvmag.png is an image showing the original vs the undersampled image for image 0 of the picture test set used in this simulation
e) summary_unetmse_150_8_2_64_1dp_cvmag_split1.txt is the model summary containing a list of layers and number of parameters, in this case for the split 1 of 5 model
f) test_pictures_Unet_MSE_150_8_2_64_1dp_CVMag.mat is the file that all of the test images fully sampled, undersampled, and reconstructed, from all 5 splits are saved to.
g) three_pic_comparison_image_0_unet_mse_150_8_2_64_1dp_cvmag_split_1.png is the picture containing the original image by the undersampled image by the reconstructed image for image 0 of the test picture set, and for the split 1 model.
h) unetmse_150_8_2_64_1dp_cvmag_split1_convergence_plot_mse_loss.png is the convergence plot for the split 1 of 5 model. Note that a) through h) involve models with the same parameters. Refer to a) to get idea how to read the parameters from the file name.


