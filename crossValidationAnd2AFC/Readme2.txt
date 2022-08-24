This folder contains two run files: 1) unet_run_crossVal.py and 2) unet_run_2AFC.py. The former file is for running the cross validation code to produce scores for kx accelerated neural
network reconstructions each from five different train-validation splits of the training data, while the latter is for training a neural network for producing kx accelerated versions of
background and signal images for 2AFC studies. Both files can only run if the same four support files and a a folder labeled "outputs" are in the same folder. Also unet_run_crossVal.py
requires that the training samples are in the directory that is specified by it, which by default is the parent folder ("../"). unet_run_2AFC.py on the other hand also requires the signal
and background image files to be contained in the specifed directory, which by default is the parent directory as well.

Below is a list of the support (or utility) files:

"data_post_processing_utils.py"
"data_prep_utils.py"
"sampling_utils.py"
"unet_utils.py"

A list of the provided data files is as follows:
"training_sample.mat"  (for training the neural network)
"backgroundImages_sample.mat" (non-accelerated test images for creating accelerated background images)
"signalImages_sample.mat" (non-accelerated test images with signal for creating accelerated signal images)

The output folder "outputs" is the folder that output files containing accelerated images, scores, etc. are sent to when running either of the two run files.

1)
Outputs of file unet_run_crossVal.py which runs cross validation on the U-Net: All outputs are sent to the "outputs" folder.
a) average_loss_calc_Unet_MSE_150_8_3_64_1dp_CVMag.mat stores the ssim (structural similarity), nmse (normalized mean squared error), nrmse (normalized root mean squared error) and norml1 (normalized l1 norm error) scores, means, and standard deviations for a U-Net with MSE loss, 150 epochs, batch size of 8,
undersampling factor k=3 (kx sampling), 64 initial filters, and dropout rate of 0.1 (1dp), and the label CVMag for example indicates that we were using cross validation on a magnitude image training set.
"Unet" is indicative that we are calling our U-Net model "Unet" in the file save names. 
Note that a) through h) involve models with the same parameters.
b) average_loss_calc_UnetMSE_150_8_3_64_1dp_CVMag.txt is the same as the mat file but everything is in ".txt" format
c) model_unetmse_150_8_3_64_1dp_cvmag_split1.h5 is a saved trained neural network model run for the first split of 5 fold cross validation.
d) original_vs_undersampled_image_0_unet_mse_150_8_3_64_1dp_cvmag.png is an image showing the original vs the undersampled image for image 0 of the picture test set used in this simulation
e) summary_unetmse_150_8_3_64_1dp_cvmag_split1.txt is the model summary containing a list of layers and number of parameters, in this case for the split 1 of 5 model
f) test_pictures_Unet_MSE_150_8_3_64_1dp_CVMag.mat is the file that all of the test images fully sampled, undersampled, and reconstructed, from all 5 splits are saved to.
g) three_pic_comparison_image_0_unet_mse_150_8_3_64_1dp_cvmag_split_1.png is the picture containing the original image by the undersampled image by the reconstructed image for image 0 of the test picture set, and for the split 1 model.
h) unetmse_150_8_3_64_1dp_cvmag_split1_convergence_plot_mse_loss.png is the convergence plot for the split 1 of 5 model. Note that a) through h) involve models with the same parameters. Refer to a) to get idea how to read the parameters from the file name.

2)
Outputs of file unet_run_2AFC.py which trains a U-Net neural network to create 2AFC files: All outputs are sent to the outputs folder.
a) model_unetmse_150_8_3_64_1dp_2afc.h5 is the h5 file that stores the trained neural network model having mse loss, 1 epoch, a batch size of 1, acceleration of 2x, 4 initial filters, and 1dp means dropout rate of 0.1. unified diagram is the unet name and singlemagartemis is the task label, indicating that this is a single non-complex valued run instead of part of cross validation and that artemis was used as the computer.
The same parameters used in 1) are used for a-g) in 2).
b) original_vs_undersampled_image_0_unet_mse_150_8_3_64_1dp_2afc.png is the picture file showing image 0 of the picture set original and undersampled side by side.
c) recon_backgroundImages_sample_unet_MSE_150_8_3_64_1dp_2AFC.mat is the reconstructed 2AFC background image file, reconstructed using a unet having the same parameters listed in a)
d) recon_signalImages_sample_unet_MSE_150_8_3_64_1dp_2AFC.mat is the same as c) but for signal images
e) summary_unetmse_150_8_3_64_1dp_2afc.txt is a model sumary of the unet specified in a)
f) test_pictures_unet_MSE_150_8_3_64_1dp_2AFC.mat is the mat file containing the test images before and after reconstruction.
g) three_pic_comparison_image_0_unet_mse_150_8_3_64_1dp_2afc.png is an image containing the original, undersampled and reconstructed versions of image 0 from the picture set
