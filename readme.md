Stability Margins of Neural Network Controllers
===================================

This repository contains code for the paper "Stability Margins of Neural Network Controllers"

## File Structure

* `envs`: plant models.
* `models`: controller models. The `DissipativeSimplestRINN` is the controller model used. See `train_controller.py` for a configuration example.
* `trainers.py`: trainers modified to include the projection step.

### Runnable files
* `train_controller.py`: configure and train controllers.

## Package Requirements

This code is tested with Python 3.10.10 and PyTorch 1.11.

## Setup

* `poetry install`
* Above will install some items then error. 
* Run `poetry run pip install setuptools==65.5.0 pip==21`
* Run `poetry run pip install wheel==0.38.0`
* Run `poetry run pip install gym==0.21`
* Now run `poetry install` again 

Then, controller training can be run with `poetry run python train_controller.py`