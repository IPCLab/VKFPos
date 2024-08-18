# VKFPos

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
Change the `--checkpoint_file` in the visualize.py to your trained checkpoint, and run `visualize.py`# VKFPos
# VKFPos
