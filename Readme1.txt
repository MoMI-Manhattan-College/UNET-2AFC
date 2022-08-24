The purpose of this folder and its contents is to train neural networks to simulate accelerated MRI for the purpose of testing a new metric based on the two alternative forced choice task, or 2AFC, for evaluating several different
acceleration schemes based on how easy it is to perform a signal detection task on a set of accelerated images. To do this two different paradigms of evaluating such acceleration schemes are included: 
using image metrics such as MSE and SSIM, and using the percent correct score from people who perform 2AFC trials on signal and background images that were accelerated. The first kind involves performing 5-fold cross validation
and averaging the NRMSE (normalized root mean squared error) and SSIM (structural similarity) across the five splits. The second kind involves training a neural network using all provided training data and then 
using the neural network to accelerate two test sets, where one is referred to as a background set because it contains images with no added signal, (the signal usualy being a bright dot). The second is referred to a
signal set. After both sets are produced, using seperate code that will be supplied or referenced at a later time, they will be randomly combined using just small patches centered about the signal or where the signal would be. The combined file will then be used to perform 2AFC
trials, having a person going through some set number of them, each time determining which image contains the signal or dot in its center. The percentage of times answered correctly determines the percent correct score. It is better to
have multiple people peform this test and to perform this test under regular conditions such as the same monitor, lighting, distance from the monitor, and after some training by performing a 2AFC trial on images with the same condition
several times until their performance levels off, prior to performing 2AFCs on other conditions, where condition could mean different kx acceleration or loss function.

This folder contains three data files, one yaml file called UNET.yml, and one subfolder called crossValidationAnd2AFC. The data files are "training_sample.mat", "backgroundImages_sample.mat", and "signalImages_sample.mat."
The first of these files is used for training the neural network, while the last two is used for creating accelerated background and signal images from a trained U-Net.

Note that the data files should be kept in the same folder as the subfolder, crossValidationAnd2AFC.

The subfolder contains all neural network related utility files and runnable scripts to run as well. It also contains an "outputs" folder 
as a folder to collect outputs produced by the run files. These are descirbed by the "Readme2.txt" file stored within the subfolder.

Note that the code in the subfolder runs using modules including TensorFlow and TensorFlow Addons.
To set up the anaconda environment to use this code, use the code 
"conda env create -f UNET.yml" to create it using the settings from the yaml file.
To activate this environment you can use "conda activate env2".
"conda deactivate" can be used to return to the base environment.

