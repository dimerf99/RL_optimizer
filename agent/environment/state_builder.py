"""
RL Environment Component: State Builder.

This module constructs the agent's state representation:
- Generates training trajectories of PINN models
- Trains an AutoEncoder on these trajectories
- Builds the loss landscape representation
- Returns a final encoded state for the RL agent.
"""

from tedeous.model import Model

from tedeous.optimizers import Optimizer
from tedeous.callbacks import EarlyStopping
from landscape_visualization._aux.plot_loss_surface import PlotLossSurface


class TrajectoryBuilder:
    def __init__(self, problem_config, rl_agent_params):
        self.problem_config = problem_config
        self.rl_agent_params = rl_agent_params

    def check_trajectories(self, trajectories, check_params):
        return len(trajectories) < check_params["n_save_models"]

    def build(self, params):
        pinn_model = Model(*self.problem_config)
        pinn_model.compile(*self.problem_config)
        pinn_model.train(
            params["optimizer"],
            params["epochs"],
            n_save_models=self.rl_agent_params['n_save_models'],
            stuck_threshold=self.rl_agent_params['stuck_threshold']
        )
        return pinn_model.min_loss, pinn_model.net


class AutoEncoderTrainer:
    def __init__(self, AE_train_params, visualization_model):
        self.AE_train_params = AE_train_params
        self.visualization_model = visualization_model

    def build(self, trajectories):
        optimizer = Optimizer('RMSprop', {'lr': self.AE_train_params['learning_rate']},
                              cosine_scheduler_patience=self.AE_train_params['cosine_scheduler_patience'])
        cb_es = EarlyStopping(patience=self.AE_train_params['patience_scheduler'])

        AE_model = self.visualization_model.train(
            optimizer, self.AE_train_params['epochs'], self.AE_train_params['every_epoch'],
            self.AE_train_params['batch_size'], self.AE_train_params['resume'],
            callbacks=[cb_es], solver_models=trajectories, finetune_AE_model=self.AE_train_params["finetune_AE_model"]
        )
        return AE_model


class LossLandscapeBuilder:
    def __init__(self, loss_surface_params, problem_config):
        self.problem_config = problem_config
        self.loss_surface_params = loss_surface_params

    def build(self, trajectories, AE_model):
        self.loss_surface_params['solver_models'] = trajectories
        self.loss_surface_params['AE_model'] = AE_model

        plot_loss_surface = PlotLossSurface(**self.loss_surface_params)
        raw_states_dict = plot_loss_surface.save_equation_loss_surface(
            self.problem_config["u_exact_test"], self.problem_config["grid_test"], self.problem_config["grid"],
            self.problem_config["domain"], self.problem_config["equation"], self.problem_config["boundaries"],
            self.problem_config["PINN_layers"]
        )

        return raw_states_dict


class StateBuilder:
    def __init__(self, trajectory_builder, AE_trainer, loss_landscape_builder):
        self.trajectory_builder = trajectory_builder
        self.AE_trainer = AE_trainer
        self.loss_landscape_builder = loss_landscape_builder

    def build(self, action):
        loss, solver_models = self.trajectory_builder.build(action)
        AE_model = self.AE_trainer.train(solver_models)
        state = self.loss_landscape_builder.build(solver_models, AE_model)
        return state
