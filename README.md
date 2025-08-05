# RL_optimizer
Reinforcement Learning-based optimizers chain selection for Physics-Informed Neural Networks (PINNs). Modular architecture with separate RL environment, state builder, reward strategies, and training workflow.

This repository provides an RL-driven framework for dynamically selecting optimizers and their hyperparameters during the training of Physics-Informed Neural Networks (PINNs).
The goal is to improve convergence and efficiency of solving differential equations by automating optimizer scheduling based on the current state of the loss landscape.

## Key Features

### Modular Design – Clear separation of RL components:

- **wrapper.py** – RL training workflow and integration with TEDEouS solver (https://github.com/ITMO-NSS-team/torch_DE_solver).

- **algorithm.py** - RL algorithm (DQN-based agent)

- **environment.py** – RL environment for optimizer selection. RL environment contains several components:

  - **state_builder.py** – Construction of environment state from PINN trajectories, AutoEncoder, and loss landscape.

  - **reward_calculator.py** – Multiple reward strategies (absolute, diff, balanced, curiosity, stagnation penalty).

  - **termination_checker.py** – Flexible done-condition evaluation.

### Plug-and-Play architecture – Easy to extend with new:

- Reward calculation strategies

- State representations

- Termination rules

**Trajectory-based State Representation** – Uses saved models to construct a loss landscape.

**Supports multiple optimizers** – RL agent dynamically chooses optimizers, learning rates, and training epochs.

**Compatible with TEDEouS solver** – Uses its training loop without redundant reimplementation.
