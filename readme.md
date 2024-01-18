Synthesis of Stabilizing Recurrent Equilibrium Network Controllers
==================================================================

This is an implementation of the stabilizing REN controller presented at CDC 2022:
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

## File Structure

* `envs`: models of plants.
* `models`: controller models and an implicit model for system identification. The controller presented in the paper is in `ProjREN.py`.
* `learned_models`: parameters trained with an implicit model for system identification.
* `activations.py`: activation functions with sector-bound information.
* `trainers.py`: trainers modified to include the projection step.

### Runnable files
* `train_controller.py`: configure and train controllers.
* `train_implicit_network.py`: train an implicit model to learn plant dynamics.
* `plots.py` and `rollout.py`: plotting files.

## Package Requirements

This code is tested with Python 3.9 and PyTorch 1.10.

## Credits
* The fixed point solvers in the `deq_lib` folder and the basis for the PyTorch implicit model implementation are from the [Deep Equilibrium Models](https://github.com/locuslab/deq) repository.

## Setup

* `poetry install`
* Above will install some items then error. Run `poetry run pip install gym==0.21`
* Now run `poetry install` again 

Then, controller training can be run with `poetry run python train_controller.py`
