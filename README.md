# JMOD2
J-MOD2: Joint Monocular Obstacle Detection and Depth Estimation

Test and training code of the paper: Mancini, Michele, et al. "J-MOD $^{2} $: Joint Monocular Obstacle Detection and Depth Estimation." International Conference on Robotics and Automation (ICRA) 2018.


[Link to paper](http://www.sira.diei.unipg.it/supplementary/jmod2_ral2018/JMOD2.pdf "Paper PDF")

# Installation

OS: Ubuntu 16.04 (other versions are not tested)

Requirements:
1. Keras 2.0 (with Tensorflow backend)
2. Python 2.7
3. OpenCV (tested on version 3.3, older versions SHOULD work as well)

For ROS node only:
1. ROS Kinetic

# Models

Training and test code is provided for J-MOD2 and the baselines cited in the paper

J-MOD2 trained weights on the UnrealDataset can be downloaded [here](http://www.sira.diei.unipg.it/supplementary/jmod2_ral2018/jmod2.hdf5)

Baselines trained weights on the UnrealDataset (depth-only, detector, Eigen, Full-MAE, JRN) can be downloaded from [here](http://www.sira.diei.unipg.it/supplementary/jmod2_ral2018/jmod2_baselines.tar.gz)
[NOTE: JRN code will be provided soon in a different repository]
The code expects to find the files in a folder called 'weights' inside the J-MOD2 root directory.

# Usage: testing on the UnrealDataset

First, download the UnrealDataset from [here](https://isar.unipg.it/index.php?option=com_content&view=article&id=53:unrealdataset&catid=17&Itemid=212) and unzip it in a folder called "data" inside the J-MOD2 root directory. 
If you need to store the dataset in a different destination, you need to change the `data_set_dir` parameter in the `config.py` file accordingly.

Then run
```
cd [J-MOD2 root dir]
evaluate_on_unrealdataset.py
```

To test different models, edit the variable `model_name` in the script. Refer to the inline comments for further indications. 

# Usage: testing on the Zurich Forest Dataset

You can download the dataset from [here](http://www.sira.diei.unipg.it/supplementary/ral2016/zurich_test_set.tar.gz). A subset with manually labelled ground truth bounding boxes for obstacles can be downloaded [here](http://www.sira.diei.unipg.it/supplementary/jmod2_ral2018/zurich_forest_dataset_with_obs_label.npy). Again, by default the files should be unzipped in a folder called "data" inside the J-MOD2 root directory.

Then run
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

The `train.py` performs training on the UnrealDataset according to the parameters in the `config.py` file.
You can train different models by simply editing the line `model = JMOD2(config)` accordingly.

If you wish to train the model on a different dataset, you will probably need to define
new classes in the `Dataset.py`,`SampleType.py`,`DataGenerationStrategy.py` and `Sequence.py` files located in
the `lib` directory to correctly process the data. The model expects RGB inputs 
and depth maps with equal resolution as target. Obstacles labels are provided as 
described [here](https://isar.unipg.it/index.php?option=com_content&view=article&id=53:unrealdataset&catid=17&Itemid=212).
We provide a script called `create_obstacles_annotations.py` in the `utils` directory to create
labels from segmentation and depth ground truth and store them in txt files (you wiil
need to edit some hardcoded values).

Feel free to report any issue, I'm sure there are :). 

Contact: michele.mancini1989@gmail.com
