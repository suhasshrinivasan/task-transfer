# Task Transfer

Code for the paper “Towards zero-shot adaptation of predictive models of neurons encoding posterior probability” (Suhas Shrinivasan, Ralf M. Haefner, Fabian H. Sinz, Edgar Y. Walker).

This repository contains a basic implementation of the task-transfer experiments and supporting utilities for the project.

## Quick start (Docker)

The Docker setup contains all dependencies. Build and start the environment with:

```bash
docker compose -d --build
```

## Project structure

- [task_transfer](task_transfer): Core package for sampling models, evaluation, and utilities.
- [experiments](experiments): Experiment scripts and notebooks.

## Notes

The code is organized to support zero-shot adaptation experiments on neural predictive models.

For paper plots, see [experiments/plotting/4n.ipynb](experiments/plotting/4n.ipynb).

For the main training code, look in [experiments/learning](experiments/learning):

- [train_likelihood.py](experiments/learning/train_likelihood.py)
- [train_flow_prior.py](experiments/learning/train_flow_prior.py) for the prior training script
- [train_posterior.py](experiments/learning/train_posterior.py)
- [train_sysident.py](experiments/learning/train_sysident.py)

The matching DataJoint entry points are the [dj_*.py](experiments/learning) scripts in the same folder.

DataJoint is used to administer and manage the training and evaluation experiments in this project. For an overview of the linked tables, see [experiments/dj/dj_interactive.ipynb](experiments/dj/dj_interactive.ipynb).
