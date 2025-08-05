import torch
import itertools
import gym
import datetime

from typing import Union, List

from tedeous.RL_repo.algorithms import DQNAgent
from tedeous.RL_repo.environment import EnvWrapper
from tedeous.callbacks.callback_list import CallbackList
from tedeous.device import device_type
from tedeous.RL_repo.utils import get_state_shape
from tedeous.utils import exact_solution_data


class Wrapper(gym.Wrapper):
    def __init__(self,
                 optimizers: dict,
                 problem_config: dict,
                 rl_agent_params: dict,
                 AE_model_params: dict,
                 AE_train_params: dict,
                 loss_surface_params: dict):
        """
        Args:
            problem_config (dict): parameters of PDE (domain, boundaries, equation, net, exact_func)
            optimizers (dict): optimizers with parameters for PINN train.
            rl_agent_params (dict): dictionary with rl agent parameters.
            AE_model_params (dict): parameters of autoencoder model.
            AE_train_params (dict): parameters of autoencoder train process.
            loss_surface_params (dict) parameters for loss surface generation.
        """
        self.optimizers = optimizers
        self.problem_config = problem_config
        self.optimizer = optimizers
        self.AE_model_params = AE_model_params
        self.AE_train_params = AE_train_params
        self.loss_surface_params = loss_surface_params
        self.rl_agent_params = rl_agent_params

        self.env = EnvWrapper(problem_config=problem_config,
                              AE_model_params=AE_model_params,
                              AE_train_params=AE_train_params,
                              loss_surface_params=loss_surface_params,
                              n_save_models=rl_agent_params['n_save_models'],
                              tolerance=rl_agent_params["tolerance"])

        # These objects must be created after the first optimizer is started
        n_observation = self.env.observation_space
        # state_dim = np.prod(env.observation_space.shape)
        n_action = self.env.action_space

        self.state_shape = get_state_shape(loss_surface_params)

        self.rl_agent = DQNAgent(n_observation,
                                 n_action,
                                 optimizer_dict=self.optimizers,
                                 memory_size=rl_agent_params["rl_buffer_size"],
                                 device=device_type(),
                                 batch_size=rl_agent_params["rl_batch_size"])

    def step(self,
             epochs: int,
             callbacks: Union[List, None] = None):

        self.saved_models = []
        self.prev_to_current_optimizer_models = []
        self.rl_penalty = 0
        self.stop_training = False

        done = None
        idx_traj = 0
        n_steps = 0
        n_steps_max = 1512
        bufer_start_i = 128  # 128
        n_steps_for_optim = 16  # n steps optimize

        while n_steps < n_steps_max:
            self.net.apply(self.reinit_weights)
            self.solution_cls._model_change(self.net)
            self.t = 1
            callbacks.set_model(self)

            # state = torch init -> AE_model
            callbacks.callbacks[0]._stop_dings = 0
            total_reward = 0
            optimizers_history = []
            prev_reward = -1
            state = {"loss_total": torch.zeros(self.state_shape),
                     "loss_oper": torch.zeros(self.state_shape),
                     "loss_bnd": torch.zeros(self.state_shape)}
            print('\n############################################################################' +
                  f'\nStarting trajectory {idx_traj + 1}/{self.rl_agent_params["n_trajectories"]} ' +
                  'with a new initial point.')

            for i in itertools.count():
                n_steps += 1
                action, action_raw, is_model = self.rl_agent.select_action(state)

                action_raw[2]['epochs'] = action_raw[1]
                action_raw = (action_raw[0], action_raw[2])
                print(f"\naction = {action}")

                # TrajectoryBuilder ############
                # optimizer = Optimizer(action['type'], action['params'])
                # self.optimizer = optimizer.optimizer_choice(self.mode, self.net)
                # closure = Closure(mixed_precision, self).get_closure(optimizer.optimizer)

                self.t = 1

                print('\n===========================================================================\n' +
                      f'\nRL agent training: step {i + 1}.'
                      f'\nTime: {datetime.datetime.now()}.'
                      f'\nUsing optimizer: {action["type"]} for {action["epochs"]} epochs.'
                      f'\nTotal Reward = {total_reward}.\n')

                # TrajectoryBuilder ##################
                # loss, solver_models = execute_training_phase(
                #     action["epochs"],
                #     n_save_models=rl_agent_params['n_save_models'],
                #     stuck_threshold=rl_agent_params['stuck_threshold']
                # )
                # TrajectoryBuilder ##################

                if loss != loss:
                    self.rl_penalty = 0
                    break

                self.env.rl_penalty = self.rl_penalty

                # TrajectoryBuilder check errors ############
                # if solver_models is None:
                #     print("Solver models are None!!!")
                #
                # if len(solver_models) < self.rl_agent_params['n_save_models']:
                #     print(f"Current number of solver models: {len(solver_models)}. "
                #           f"\nRight number = {self.rl_agent_params['n_save_models']}")
                # TrajectoryBuilder check errors ############

                # reward parts calculation ########################
                # net = self.net.to(device_type())
                #
                # if callable(self.rl_agent_params["exact_solution"]):
                #     operator_rmse = torch.sqrt(
                #         torch.mean((self.rl_agent_params["exact_solution"](grid).reshape(-1, 1) - net(grid)) ** 2)
                #     )
                # else:
                #     exact = exact_solution_data(grid, self.rl_agent_params["exact_solution"],
                #                                 self.problem_config[-1][0], self.problem_config[-1][-1],
                #                                 t_dim_flag='t' in list(self.domain.variable_dict.keys()))
                #     net_predicted = net(grid)
                #     operator_rmse = torch.sqrt(torch.mean((exact.reshape(-1, 1) - net_predicted) ** 2))
                #
                # boundary_rmse = torch.sum(torch.tensor([
                #     torch.sqrt(torch.mean((bconds[i]["bval"].reshape(-1, 1) - net(bconds[i]["bnd"])) ** 2))
                #     for i in range(len(bconds))]))
                #
                # self.env.solver_models = solver_models
                # self.env.reward_params = {
                #     "operator": {
                #         "error": operator_rmse,
                #         "coeff": self.rl_agent_params["reward_operator_coeff"]
                #     },
                #     "bconds": {
                #         "error": boundary_rmse,
                #         "coeff": self.rl_agent_params["reward_boundary_coeff"]
                #     }
                # }
                # reward parts calculation ########################

                optimizers_history.append(action["type"])
                print(f'\nPassed optimizer {action["type"]}.')

                # input weights (for generate state) and loss (for calculate reward) to step method
                # first getting current models and current losses
                next_state, reward, done, _ = self.env.step()

                # RewardCalculator #################
                # opt_model_i = -1
                # reward_model_i = -1
                # if prev_reward == -1:
                #     reward_model_i = 1 / reward * -1
                # elif is_model and prev_reward != -1:
                #     opt_model_i = self.rl_agent.opt_step
                #     reward_model_i = reward - prev_reward
                # else:
                #     reward_model_i = reward - prev_reward
                #
                # prev_reward = reward
                # reward_model_i_raw = reward_model_i
                # reward_model_i -= 0.01 * i
                #
                # if done == 1:
                #     reward_model_i += torch.tensor(100, dtype=torch.int8)
                # elif done == 0:
                #     pass
                # elif done == -1:
                #     reward_model_i -= torch.tensor(100, dtype=torch.int8)
                # RewardCalculator #################

                self.rl_agent.push_memory((state, next_state, action_raw, reward_model_i, abs(done),
                                           float(reward_model_i_raw), opt_model_i))

                if self.rl_agent.replay_buffer.__len__() >= bufer_start_i and \
                        self.rl_agent.replay_buffer.__len__() % n_steps_for_optim == 0:
                    print(f'\n[{datetime.datetime.now()}] RL agent optimization step {self.rl_agent.opt_step + 1}.')
                    self.rl_agent.optim_()
                    self.rl_agent.render_Q_function()
                    done = -1

                state = next_state
                total_reward += reward

                print(f'\nCurrent reward after {action["type"]} optimizer: {reward}.\n'
                      f'Total reward after using {", ".join(optimizers_history)} '
                      f'{"optimizers" if len(optimizers_history) > 1 else "optimizer"}: {total_reward}.\n'
                      f'\ndone = {done}')

                callbacks.callbacks[1].save_every = self.t
                self.env.render()

                if done == 1:
                    break
                elif done == 0:
                    continue
                elif done == -1:
                    self.rl_penalty = 0
                    break

            if done == 1:
                idx_traj += 1
