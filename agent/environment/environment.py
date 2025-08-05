import gym
import matplotlib.pyplot as plt

from typing import List, Union

from tedeous.callbacks.callback_list import CallbackList
from tedeous.RL_repo.state_builder import TrajectoryBuilder, AutoEncoderTrainer, LossLandscapeBuilder, StateBuilder
from tedeous.RL_repo.reward_calculator import RewardCalculator
from tedeous.RL_repo.termination_checker import TerminationChecker

from landscape_visualization._aux.visualization_model import VisualizationModel


class EnvWrapper(gym.Env):
    def __init__(self,
                 optimizers: dict = None,
                 problem_config: dict = None,
                 loss_surface_params: dict = None,
                 rl_agent_params: dict = None,
                 AE_model_params: dict = None,
                 AE_train_params: dict = None,
                 reward_method: str = "absolute",
                 callbacks: Union[CallbackList, List, None] = None,
                 n_save_models: int = None,
                 tolerance: float = 1e-2):
        super(EnvWrapper, self).__init__()

        self.solver_models = None
        self.reward_params = None
        self.rl_penalty = 0
        self.raw_states_dict = {}

        self.rl_agent_params = rl_agent_params
        self.AE_model_params = AE_model_params
        self.AE_train_params = AE_train_params
        self.loss_surface_params = loss_surface_params
        self.problem_config = problem_config
        self.reward_method = reward_method
        self.callbacks = callbacks

        self.visualization_model = VisualizationModel(**self.AE_model_params)
        self.plot_loss_surface = None

        self.action_space = {key: len(value) for key, value in optimizers.items()}
        self.observation_space = self.visualization_model.latent_dim + 1

        self.rl_agent = None
        self.current_reward = None
        self.done = None

        self.reward_history = []
        self.tolerance = tolerance
        self.counter = 1
        self.n_save_models = n_save_models

    def reset(self):
        """Reset environment - load error surface, reset history to zero, select starting point."""
        self.current_reward = self.reward_history[-1]
        self.counter += 1

    def step(self, action):
        """Applying an action (optimizer selection) and updating the state."""
        action, action_raw, is_model = action

        trajectory_builder = TrajectoryBuilder(self.problem_config, self.rl_agent_params)
        AE_trainer = AutoEncoderTrainer(self.AE_train_params, self.visualization_model)
        loss_landscape_builder = LossLandscapeBuilder(self.loss_surface_params, self.problem_config)
        state_builder = StateBuilder(trajectory_builder, AE_trainer, loss_landscape_builder)

        reward_calculator = RewardCalculator(self.reward_history, self.reward_method, self.reward_params, self.rl_agent)
        termination_checker = TerminationChecker(self.current_reward, self.tolerance, self.rl_penalty)

        state = state_builder.build(action)
        reward = reward_calculator.step(self.rl_penalty, is_model, self.counter)
        done = termination_checker.check()

        return state, reward, done, {}

    def render(self):
        """Display the current error and convergence history."""

        self.reset()

        # print(f"Optimizer: {self.current_optimizer['name']}, Loss: {self.current_loss}")

        # Plotting PDE solution
        self.callbacks.on_epoch_end()
        self.callbacks.callbacks[1].save_every = 0.1

        # Plotting loss landscape
        self.plot_loss_surface.plotting_equation_loss_surface(**self.problem_config)

    def close(self):
        plt.close('all')
