May 28 2021
By Joshua Herman
Code Written by Joshua Herman and Rachel Roca

This folder is "oneruntestsplitmag052721-artemis". It contains all of the latest version of single neural network run and 2AFC reconstructed background and signal producing code for the unet. This is the Joshua Herman version of the code. Each of these files contain documentation
such as docstrings at the top of the file that gives an idea of what the file is about, and possibly some advice or examples, particularly on the neural net run files.

This folder contains 3 files which can be immediately run, and the rest which are utilities containing supporting functions.
These runnable files are the following:

1) unet_run_mag_110220.py is the file set up to train and run a unet primarily for the purpose of getting reconstructions for the 2AFC trials. No validation information like scores or cross validation plots are produced.
It is currently set up for the small dataset '/FLAIRVolumesSOS' on my account jherman on the Artemis Computer. The code can be adapted to work with the large dataset as well. You may want to change certain parameters 
in the file such as the directory, file name of the fully sampled mri images, the hyperparameters of the neural network and undersampling, and the qualifier strip that ends up in output filenames. Most importantly are the hyperparameters, kx acceleration level, and qualifier components that are shown in In[2].
You must change these to specify what kind of network you want to run and how you want the files' output from the code to display the associated information in their name. This file is very similar to the code used to produce the 2AFC files for the abstract, there are no differences in this file or in the support function files that should make this code
perform any different.  Note the docstring at the top of the file that mentions the kinds of things that you might want to change in the file between runs. All outputs are saved to folder temp_single, which should be in the same directory as this file.

Outputs of unet_run_mag_110220.py: All outputs are sent to the temp_single folder. If you don't have this folder in the run file's directory, be sure to create one.
a) model_unified_diagrammse_1_1_2_4_1dp_singlemagartemis.h5 is the h5 file that stores the trained neural network model having mse loss, 1 epoch, a batch size of 1, acceleration of 2x, 4 initial filters, and 1dp means dropout rate of 0.1. unified diagram is the unet name and singlemagartemis is the task label, indicating that this is a single non-complex valued run instead of part of cross validation and that artemis was used as the computer.
These parameters can be read from any file name as (beginning of file name)_loss_epochs_(batch size)_acceleration_(initial filters)_ dropout_(task label)_ (end of file name).
b) original_vs_undersampled_image_0_unified_diagram_mse_1_1_2_4_1dp_singlemagartemis.png is the picture file showing image 0 of the picture set original and undersampled side by side.
c) recon_FLAIR_NN_testing_backgroundImages_11_22_2020_Unified Diagram_MSE_1_1_2_4_1dp_SingleMagArtemis is the reconstructed 2AFC background image file, reconstructed using a unet having the same parameters listed in a)
d) recon_FLAIR_NN_testing_signalImages_11_22_2020_Contrast5e-5_Unified Diagram_MSE_1_1_2_4_1dp_SingleMagArtemis is the same as c) but for signal images
e) summary_unified_diagrammse_1_1_2_4_1dp_singlemagartemis.txt is a model sumary of the unet specified in a)
f) test_pictures_Unified Diagram_MSE_1_1_2_4_1dp_SingleMagArtemis.mat is the mat file containing the test images before and after reconstruction.
g) three_pic_comparison_image_0_unified_diagram_mse_1_1_2_4_1dp_singlemagartemis is an image containing the original, undersampled and reconstructed versions of image 0 from the picture set

2) unet_run_single_val_030321.py is the file set up to train and run a unet both for the purpose of getting reconstructions for the 2AFC trials and for validating the neural network by scoring it on a test set and getting a convergence plot using test set data. It is currently set up for the large dataset on my account jherman on the Artemis Computer.
One big difference is that multiple files have output combined to get the training set data, and testing set data file is also used to get sample picture test output as opposed to taking the data from the training set file.
Note the docstring at the top of this file, which mentions some areas of the file you might want to change between runs. Most importantly are the hyperparameters, kx acceleration level, and qualifier components that are shown in In[2].
You must change these to specify what kind of network you want to run and how you want the files' output from the code to display the associated information in their name. All outputs are saved to folder temp_single, which should be in the same directory as this file.

Outputs of unet_run_single_val_030321.py: All outputs are sent to the temp_single folder. If you don't have this folder in the run file's directory, be sure to create one.
a) average_loss_calc_Unified Diagram_SSIM_1_1_2_4_1dp_SingleValMag-Artemis.mat is the mat file containing the SSIM, NMSE, NRMSE, and NORML1 scores for the model having parameters decoded into the file name, as described in 1) a).
b) average_loss_calc_Unified DiagramSSIM_1_1_2_4_1dp_SingleValMag-Artemis.txt is the text file containing the same information
c) model_unified_diagramssim_1_1_2_4_1dp_singlevalmag-artemis.h5 is the model file containing the trained unet
d) original_vs_undersampled_image_0_unified_diagram_ssim_1_1_2_4_1dp_singlevalmag-artemis.png is image 0 from the picture set both original and undersampled side by side
e) recon_FLAIR_NN_testing_backgroundImages_11_22_2020_Unified Diagram_SSIM_1_1_2_4_1dp_SingleValMag-Artemis.mat is the reconstructed background images
f) recon_FLAIR_NN_testing_signalImages_11_22_2020_Contrast5e-5_Unified Diagram_SSIM_1_1_2_4_1dp_SingleValMag-Artemis.mat is the reconstructed signal images
g) summary_unified_diagramssim_1_1_2_4_1dp_singlevalmag-artemis.txt is a summary of the unet mentioning every layer and number of parameters per layer and output shape of layer and connectivity
h) test_pictures_Unified Diagram_SSIM_1_1_2_4_1dp_SingleValMag-Artemis.mat contains the test images before and after reconstruction
i) three_pic_comparison_image_0_unified_diagram_ssim_1_1_2_4_1dp_singlevalmag-artemis.png is test picture 0 original image by undersampled image by reconstructed image compared side by side
j) unified_diagramssim_1_1_2_4_1dp_singlevalmag-artemis_convergence_plot_one_minus_ssim_loss.png is the convergence plot for the model.


3) score_images_from_saved_network.py scores a neural network on a dataset using 4 metrics: SSIM, NRMSE, NMSE, and NORML1. It does so by loading the trained unet model from its h5 file and a picture dataset from its mat file and using the function score_images_using_model_and_images_from_file which is defined in that file to get the scores. This function
could alternatively be loaded several times for several dataset network combinations after being imported to another file. The scores are saved to a text file in the folder temp_additional_scores, which should be in the same directory as this file.

a) loss_calc_model_unified_diagramssim_1_1_2_4_1dp_singlevalmag-artemis.h5_FLAIR_NN_testing_backgroundImages_11_22_2020_2.txt is the text file containing the score results.


The main folder also has 4 utility files that aren't runnable but must be in the same folder as the runnable files, at least for files 1) and 2).

4) data_post_processing_110220.py contains the functions for plotting the convergence plots and the code for displaying the fully sampled image besides the undersampled image beside the reconstructed image.

5) data_prep_utils_110220.py contains the functions for normalizing the images, displaying a fully sampled image besides its undersampled counterpart, and loading/shuffling/normalizing the fully sampled mri images from file, and a function for splitting the data into a training set and a set for test images.

6) sampling_utils_110220.py contains the functions for creating and displaying a sampling mask as well as applying it to undersample data.

7) unet_unified_utils_110220.py functions for all things neural network and 2AFC file creation related, such as the ssim metric, the function that creates 2AFC files create_auc_images, the functions
for running a unet and producing 2AFC files and/ or validation results: single_run_val_auc and single_run_create_auc_pics, the unet and its block components



To summarize changes between this code and my abstract code (onerunmag011721), other than comments there aren't differences in data_post_processing_110220.py and data_prep_utils_110220.py and the only difference in sampling_utils_110220.py is that the newer version's display_mask function produces a plot with a higher dpi of 1200 as opposed to 100 to remove some artifacts of low sampling.
unet_unified_utils_110220.py is mostly the same but new one contains the additional function single_run_val_auc and there is a small change to how file save names are created in the create_auc_images function. File 3) was changed slightly to prevent errors and to not have a problem with loading files from other directories.
