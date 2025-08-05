"""
RL Environment Component.
Contains all classes required to determine termination conditions
for RL episodes during optimizer selection.

Handles:
- Convergence detection
- NaN/Inf loss early stopping
- Local minimum detection
- Penalization of invalid or unstable training steps
"""

import numpy as np


class TerminationChecker:
    def __init__(self, tolerance: float, max_steps: int = None, penalty_on_nan: float = -1.0):
        """
        Args:
            tolerance (float): Threshold for considering the solution converged.
            max_steps (int, optional): Maximum number of steps before forced termination.
            penalty_on_nan (float): Penalty applied when loss is NaN or diverges.
        """
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.penalty_on_nan = penalty_on_nan

    def check(self, current_loss, prev_losses=None, step=None, stop_flag=False):
        """
        Determine the done flag for the RL environment.

        Args:
            current_loss (float): Current training loss or reward.
            prev_losses (list): Recent history of losses to detect local minima or plateaus.
            step (int): Current step number in the trajectory.
            stop_flag (bool): Signal from solver about early stopping (e.g., callback trigger).

        Returns:
            done (int):
                 1 -> success (solution converged),
                -1 -> failed trajectory (divergence, NaN, forced stop),
                 0 -> continue training.
            penalty (float): Penalty value to apply to reward if done = -1.
        """
        # Check for NaN or divergence
        if np.isnan(current_loss) or current_loss == np.inf or current_loss > 1e3:
            return -1, self.penalty_on_nan

        # Check convergence (success)
        if abs(current_loss) < self.tolerance:
            return 1, 0.0

        # Detect local minimum (loss stagnation)
        if prev_losses is not None and len(prev_losses) > 5:
            if np.std(prev_losses[-5:]) < 1e-6:
                return -1, self.penalty_on_nan

        # Forced stop from solver callbacks
        if stop_flag:
            return -1, self.penalty_on_nan

        # Step limit reached
        if self.max_steps is not None and step is not None and step >= self.max_steps:
            return -1, 0.0

        # Otherwise continue
        return 0, 0.0
