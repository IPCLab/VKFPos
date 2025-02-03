# VKFPos: A Learning-Based Monocular Positioning with Variational Bayesian Extended Kalman Filter Integration

This work addresses the challenges in learning-based monocular positioning by proposing VKFPos, a novel approach that integrates Absolute Pose Regression (APR) and Relative Pose Regression (RPR) via an Extended Kalman Filter (EKF) within a variational Bayesian inference framework. Our method shows that the essential posterior probability of the monocular positioning problem can be decomposed into APR and RPR components. This decomposition is embedded in the deep learning model by predicting covariances in both APR and RPR branches, allowing them to account for associated uncertainties. These covariances enhance the loss functions and facilitate EKF integration. Experimental evaluations on both indoor and outdoor datasets show that the single-shot APR branch achieves accuracy on par with state-of-the-art methods. Furthermore, for temporal positioning, where consecutive images allow for RPR and EKF integration, VKFPos outperforms temporal APR and model-based integration methods, achieving superior accuracy.

The preprint is available at https://arxiv.org/abs/2501.18994

## Setup
Before create environment, first check

* cuda version `nvcc -V`, for pytorch version
<!-- * update conda package version `conda update --all` -->

If cuda version > 11.7, you can direct run `pip install -r requirements.txt`

If not, open `requirements.txt` and delete all related packages about pytorch, and run `pip install -r requirements.txt`, finally, install the newest pytorch you can use according to your cuda version, by the way, the test version is under pytorch version==2.0.1.

Create some folders by following commands:
```
mkdir his
mkdir log
mkdir datasets
mkdir export
```

And create a soft link to your 7Scenes or RobotCar dataset, or direct put it in datasets folder

```
cd datasets
ln -s <7Scenes-path-you-download> 7Scenes
```


## Training 
Direct Run `train.py` in the terminal.

* If you want to training on RobotCar dataset, must change the `--data_set` arg in `train.py` first.


## Visualize
Change the `--checkpoint_file` in the visualize.py to your trained checkpoint, and run `visualize.py`
