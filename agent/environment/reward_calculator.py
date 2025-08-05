"""
RL Environment Component.
Contains all classes and strategies required to compute reward signals
based on operator and boundary errors and penalties.

Available reward strategies:
- Difference-based reward
- Absolute error-based reward
- Curiosity-driven reward
- Balanced operator-boundary reward
- Dynamic penalty for repeated actions
"""

from abc import ABC, abstractmethod


class BaseRewardStrategy(ABC):
    """Basic interface for all reward strategies."""

    @abstractmethod
    def compute(self, reward_params, reward_history, rl_penalty, iteration, rl_agent, is_model):
        """
        Calculates the reward for the agent.

        Args:
            reward_params (dict): Dictionary with operator and boundary error values and coefficients.
            reward_history (list): History of previous rewards.
            rl_penalty (float): Penalty applied to reward.
            iteration (int): Current training iteration.
            rl_agent: RL agent instance for action tracking.
            is_model (bool): Indicates whether the model state is used.

        Returns:
            float: The computed reward value.
        """
        pass


class DiffRewardStrategy(BaseRewardStrategy):
    """Difference in rewards: a reward is the difference between the previous and current rewards."""

    def compute(self, reward_params, reward_history, rl_penalty, iteration, rl_agent, is_model):
        prev_reward = reward_history[-1] if reward_history else 0
        current_reward = reward_params["operator"]["coeff"] * reward_params["operator"]["error"] + reward_params[
            "bconds"]["coeff"] * reward_params["bconds"]["error"]
        reward = (prev_reward - current_reward) + rl_penalty
        return reward


class AbsoluteRewardStrategy(BaseRewardStrategy):
    """Absolute reward: simply minus the current error."""

    def compute(self, reward_params, reward_history, rl_penalty, iteration, rl_agent, is_model):
        current = reward_params["operator"]["coeff"] * reward_params["operator"]["error"] \
                  + reward_params["bconds"]["coeff"] * reward_params["bconds"]["error"]
        return -current + rl_penalty


class BalancedRewardStrategy(BaseRewardStrategy):
    """Balance between operator error and boundaries (encourages a uniform reduction of both)."""

    def compute(self, reward_params, reward_history, rl_penalty, iteration, rl_agent, is_model, coeff=1):
        op = reward_params["operator"]["error"]
        bnd = reward_params["bconds"]["error"]
        rl_penalty = coeff * abs(op - bnd)
        reward = -(op + bnd) + rl_penalty
        return reward


class CuriosityRewardStrategy(BaseRewardStrategy):
    """Encourages the agent to explore new actions (addition to the reward for diversity)."""

    def __init__(self):
        self.action_memory = {}

    def compute(self, reward_params, reward_history, rl_penalty, iteration, rl_agent, is_model):
        base_reward = AbsoluteRewardStrategy().compute(reward_params, reward_history, rl_penalty, iteration, rl_agent,
                                                       is_model)
        if rl_agent and hasattr(rl_agent, 'last_action'):
            action = rl_agent.last_action
            self.action_memory[action] = self.action_memory.get(action, 0) + 1
            if self.action_memory[action] > 5:
                base_reward -= 5
        return base_reward


class StagnationPenaltyRewardStrategy(BaseRewardStrategy):
    """Penalty for lack of progress in the reward."""

    def compute(self, reward_params, reward_history, rl_penalty, iteration, rl_agent, is_model):
        prev_reward = reward_history[-1] if reward_history else 0
        reward = AbsoluteRewardStrategy().compute(reward_params, reward_history, rl_penalty, iteration, rl_agent,
                                                  is_model)
        if len(reward_history) > 10 and abs(reward - prev_reward) < 1e-5:
            reward -= 10
        return reward


class RewardCalculator:
    """
    The main class that selects and applies a reward calculation strategy
    during reinforcement learning optimization of a PINN.

    STRATEGIES: Mapping of available reward strategy names to their classes.
    """
    STRATEGIES = {
        "diff": DiffRewardStrategy,
        "absolute": AbsoluteRewardStrategy,
        "balanced": BalancedRewardStrategy,
        "curiosity": CuriosityRewardStrategy,
        "stagnation": StagnationPenaltyRewardStrategy
    }

    def __init__(self, reward_method: str = "absolute"):
        """
        Args:
            reward_method (str): The name of the reward strategy to use.
                                 Must be one of STRATEGIES keys.
        Raises:
            ValueError: If the specified reward_method is unknown.
        """
        if reward_method not in self.STRATEGIES:
            raise ValueError(f"Unknown reward method: {reward_method}")
        self.strategy = self.STRATEGIES[reward_method]()
        self.reward_history = []
        self.reward_params = None

    def update_params(self, operator_rmse: float, boundary_rmse: float, coeffs: dict):
        """
        Updates the reward parameters with new loss values.

        Args:
            operator_rmse (float): Error value for the PDE operator.
            boundary_rmse (float): Error value for the boundary conditions.
            coeffs (dict): Coefficients for weighting different error components.
        """
        self.reward_params = {
            "operator": {"error": operator_rmse, "coeff": coeffs.get("operator", 1.0)},
            "bconds": {"error": boundary_rmse, "coeff": coeffs.get("bconds", 1.0)}
        }

    def step(self, rl_penalty: float, is_model: bool, iteration: int, rl_agent=None):
        """
        Computes the next reward using the selected strategy and stores it in history.

        Args:
            rl_penalty (float): Penalty to apply in case of instability or poor training.
            is_model (bool): Whether the current step corresponds to a saved model state.
            iteration (int): Current training step.
            rl_agent (optional): RL agent instance, if the strategy uses it. Default is None.

        Returns:
            float: The computed reward for this step.
        """
        reward = self.strategy.compute(self.reward_params, self.reward_history,
                                       rl_penalty, iteration, rl_agent, is_model)
        self.reward_history.append(reward)
        return reward
