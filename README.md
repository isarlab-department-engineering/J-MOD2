# JMOD2
J-MOD2: Joint Monocular Obstacle Detection and Depth Estimation

Test and training code of the paper: Mancini, Michele, et al. "J-MOD $^{2} $: Joint Monocular Obstacle Detection and Depth Estimation." International Conference on Robotics and Automation (ICRA) 2018.

# Link to paper

[Link to paper](https://arxiv.org/pdf/1709.08480.pdf "Paper PDF")

# Installation

OS: Ubuntu 16.04 (other versions are not tested)

Requirements:
1. Keras (with Tensorflow backend)
2. Python 2.7
3. OpenCV (tested on version 3.3, older versions SHOULD work as well)

For ROS node only:
1. ROS Kinetic

# Models

Training and test code is provided for J-MOD2 and the baselines cited in the paper

J-MOD2 trained weights on the UnrealDataset can be downloaded [here](http://www.sira.diei.unipg.it/supplementary/jmod2_ral2018/jmod2.hdf5)

Baselines trained weights on the UnrealDataset (depth-only, detector, Eigen, FUll-MAE, JRN) can be downloaded from [here](http://www.sira.diei.unipg.it/supplementary/jmod2_ral2018/jmod2_baselines.tar.gz)
[NOTE: JRN code is not online, it will be uploaded as soon as possible]
The code expects to find the files in a folder called 'weights' inside the J-MOD2 root directory.

# Usage: testing on the UnrealDataset

Download the UnrealDataset test data from [here](http://www.sira.diei.unipg.it/supplementary/jmod2_ral2018/UnrealDataset_Test.tar.gz) and unzip it in a folder called "data" inside the J-MOD2 root directory. [If you need to store the dataset in other destinations, you need to change the 'data_set_dir' parameter in the config.py file]

Then run
```
cd [J-MOD2 root dir]
evaluate_on_unrealdataset.py
```
If the dataset and the weights are placed into the default directories, evaluation of the J-MOD2 model should run seamlessly.
To test different models, edit the variable `model_name` in the script accordingly. Refer to the inline comments for further indications. 

# Usage: testing on the Zurich Forest Dataset

You can download the full dataset from [here](http://www.sira.diei.unipg.it/supplementary/ral2016/zurich_test_set.tar.gz). A subset with manually labelled ground truth bounding boxes for obstacles can be downloaded [here](http://www.sira.diei.unipg.it/supplementary/jmod2_ral2018/zurich_forest_dataset_with_obs_label.npy). Again, by default the files should be unzipped in a folder called "data" inside the J-MOD2 root directory.

After that, run
```
cd [J-MOD2 root dir]
evaluate_on_zurich.py #to test on the full dataset
evaluate_on_labeled_zurich.py #to test on the subset
```
# Usage on custom data

You can follow the structure of the provided evaluation codes to easily write new evaluation scripts on different datasets. Depth maps and obstacle bounding boxes are computed by the `run` method of each model's class

Example:
```
model = JMOD2(config) #or any other model
[predicted_depth, predicted_obstacles, corrected_depth] = model.run(rgb_input) 
```
The rgb input is expected to be normalized in a [0..1] range. This method returns a tuple with:
1. Predicted depth (in meters)
2. A list of the predicted obstacles. Obstacles are returned as Python objects with x,y,w,h fields (x,y=top left corner coordinates in image space, w,h = width and height)  for its bounding box description and depth_mean, depth_variance fields with the regressed depth distribution's statistics of the predicted obstacle.
3. Corrected depth (in meters, refer to the paper for technical details)

If the model does not compute some of the above-mentioned fields (for example, Eigen's model does not compute detections and corrected depth), None object are returned in the respective fields.

# ROS node

COMING SOON

# Training

UPDATE 18/01/2017: The training code is provided, although the training set of the UnrealDataset is not yet online (it will be uploaded soon). More documentation about the training code will be written soon.
