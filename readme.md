Neural Network Controller Synthesis
===================================

This repository contains code for the following papers:
* "Synthesizing Neural Network Controllers with Closed-Loop Dissipativity Guarantees".
  * Code is the `master` branch.
  * The `DissipativeSimplestRINN` with mode `thetahat` is the training method and controller model presented in this paper. See `train_controller.py` for example usage.
* "Synthesis of Stabilizing Recurrent Equilibrium Network Controllers".
  * Code is at git tag `TODO`.


## File Structure

* `envs`: plant models.
* `models`: controller models.
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

## Papers in this Repository

```
TODO
```
and
```
@INPROCEEDINGS{9992684,
  author={Junnarkar, Neelay and Yin, He and Gu, Fangda and Arcak, Murat and Seiler, Peter},
  booktitle={2022 IEEE 61st Conference on Decision and Control (CDC)}, 
  title={Synthesis of Stabilizing Recurrent Equilibrium Network Controllers}, 
  year={2022},
  volume={},
  number={},
  pages={7449-7454},
  doi={10.1109/CDC51059.2022.9992684}}
```