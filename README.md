# Yoga-Pose-Detection-Model

This project mainly aims to achieve the correct yoga pose. You can simply provide any video with yoga poses, and it can briefly tell you the name of the pose if found from the training dataset. We have used 82 classes to train this model (82 Yoga Poses). The dataset contains a total of approximately 9400+ images.

## Steps to Run this Model

1. Download the trained model named as `pose_classifier_advanced.h5` or clone the repository.
2. Run `training.py`, and while running, type `test` and also provide the location of the video (as mentioned in the code).

## Steps to Create this Model on Your Own

1. Clone the repository.
2. Create a dataset or download a dataset from [Kaggle](https://www.kaggle.com/) or any other website. (We have 82 classes, meaning 82 types of yoga poses.)
3. Make sure to modify the `pose_name.py` file. In our case, there are 82 pose names. You will need to change it according to your dataset.
