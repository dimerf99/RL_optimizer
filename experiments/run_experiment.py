import os

from tedeous.RL_repo.wave_config import get_problem_config
from tedeous.RL_repo.wrapper import Wrapper

problem_config = get_problem_config(grid_res=50)

optimizers = {
    'Adam': {
        'lr': [1e-4, 1e-4, 1e-4],
        'epochs': [100, 500, 1000]
    },
    'LBFGS': {
        'lr': [1, 5e-1, 1e-1],
        'epochs': [100, 500, 1000]
    },
    'PSO': {
        'lr': [5e-3, 1e-3, 1e-4],
        'epochs': [100, 500, 1000]
    },
    'NNCG': {
        'lr': [1, 5e-1, 1e-1, 5e-2, 1e-2],
        'epochs': [50, 51, 52],
        'precond_update_frequency': [10, 15, 20]
    }
}

AE_model_params = {
    "mode": "NN",
    "num_of_layers": 3,
    "layers_AE": [991, 125, 15],
    "num_models": None,
    "from_last": False,
    "prefix": "model-",
    "every_nth": 1,
    "grid_step": 0.1,
    "d_max_latent": 2,
    "anchor_mode": "circle",
    "rec_weight": 10000.0,
    "anchor_weight": 0.0,
    "lastzero_weight": 0.0,
    "polars_weight": 0.0,
    "wellspacedtrajectory_weight": 0.0,
    "gridscaling_weight": 0.0,
    "device": "cpu"
}

AE_train_params = {
    "epochs": 1000,
    "patience_scheduler": 400,
    "cosine_scheduler_patience": 120,
    "batch_size": 32,
    "every_epoch": 100,
    "learning_rate": 5e-4,
    "resume": True,
    "finetune_AE_model": False,
    "finetune_AE_params": {
        "epochs": 1000,
        "patience_scheduler": 400,
        "cosine_scheduler_patience": 120,
    }
}

loss_surface_params = {
    "loss_types": ["loss_total", "loss_oper", "loss_bnd"],
    "every_nth": 1,
    "num_of_layers": 3,
    "layers_AE": [991, 125, 15],
    "batch_size": 32,
    "num_models": None,
    "from_last": False,
    "prefix": "model-",
    "loss_name": "loss_total",
    "x_range": [-1.25, 1.25, 25],
    "vmax": -1.0,
    "vmin": -1.0,
    "vlevel": 30.0,
    "key_models": None,
    "key_modelnames": None,
    "density_type": "CKA",
    "density_p": 2,
    "density_vmax": -1,
    "density_vmin": -1,
    "colorFromGridOnly": True,
    "img_dir": os.path.join(os.path.dirname(__file__), '../../examples/examples_wave/wave_1d_basic_img')
}

rl_agent_params = {
    "n_save_models": 10,
    "n_trajectories": 1000,
    "tolerance": 1e-1,
    "stuck_threshold": 10,
    "rl_buffer_size": 4,
    "rl_batch_size": 32,
    "rl_reward_method": "absolute",
    "exact_solution": problem_config['exact_func'],
    "reward_operator_coeff": 1,
    "reward_boundary_coeff": 1
}

rl_wrapper = Wrapper(
    problem_config=problem_config,
    optimizers=optimizers,
    rl_agent_params=rl_agent_params,
    AE_model_params=AE_model_params,
    AE_train_params=AE_train_params,
    loss_surface_params=loss_surface_params
)

epochs = 1e5
rl_wrapper.step(epochs)
