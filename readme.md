Synthesis of Stabilizing Recurrent Equilibrium Network Controllers
==================================================================

This is an implementation of the stabilizing REN controller presented in:
```
TODO
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