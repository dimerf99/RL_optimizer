import time
import torch
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import adaptive_lambda, cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data

solver_device('cuda')
data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../PINNacle_data/poisson1_cg_data.npy"))


def get_problem_config(grid_res):
    exp_dict_list = []

    x_min, x_max = -0.5, 0.5
    y_min, y_max = -0.5, 0.5

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)

    boundaries = Conditions()

    # Circle type of removed domains ###################################################################################

    removed_domains_lst = [
        {'circle': {'center': (0.3, 0.3), 'radius': 0.1}},
        {'circle': {'center': (-0.3, 0.3), 'radius': 0.1}},
        {'circle': {'center': (0.3, -0.3), 'radius': 0.1}},
        {'circle': {'center': (-0.3, -0.3), 'radius': 0.1}}
    ]

    # Boundary conditions ##############################################################################################

    # CSG boundaries

    boundaries.dirichlet({'circle': {'center': (0.3, 0.3), 'radius': 0.1}}, value=0)
    boundaries.dirichlet({'circle': {'center': (-0.3, 0.3), 'radius': 0.1}}, value=0)
    boundaries.dirichlet({'circle': {'center': (0.3, -0.3), 'radius': 0.1}}, value=0)
    boundaries.dirichlet({'circle': {'center': (-0.3, -0.3), 'radius': 0.1}}, value=0)

    # Non CSG boundaries

    boundaries.dirichlet({'x': x_min, 'y': [y_min, y_max]}, value=1)
    boundaries.dirichlet({'x': x_max, 'y': [y_min, y_max]}, value=1)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_min}, value=1)
    boundaries.dirichlet({'x': [x_min, x_max], 'y': y_max}, value=1)

    equation = Equation()

    # Operator: -u_xx - u_yy = 0

    poisson = {
        '-d2u/dx2':
            {
                'coeff': -1.,
                'term': [0, 0],
                'pow': 1,
                'var': 0
            },
        '-d2u/dy2':
            {
                'coeff': -1.,
                'term': [1, 1],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(poisson)

    neurons = 100

    net = torch.nn.Sequential(
        torch.nn.Linear(pde_dim_in, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, pde_dim_out)
    )

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    mode = "autograd"
    lambda_operator, lambda_bound = 1, 100

    model_layers = [2, neurons, neurons, 1]
    grid = domain.build('NN').to('cuda')
    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
    u_exact_test = exact_func(grid_test).reshape(-1)
    loss_surface_equation_params = [u_exact_test, grid_test, grid, domain, equation, boundaries, model_layers]

    img_dir = os.path.join(os.path.dirname(__file__), 'wave_1d_basic_RL_img')
    cb_cache = Cache(cache_verbose=True, model_randomize_parameter=1e-6)
    cb_es = EarlyStopping(eps=1e-6,
                          loss_window=100,
                          no_improvement_patience=100,
                          patience=20,
                          randomize_parameter=1e-4,
                          info_string_every=1)

    mixed_precision = False
    callbacks = [cb_cache, cb_es]
    models_concat_flag = False

    return {
        'domain': domain,
        'conditions': boundaries,
        'equation': equation,
        'net': net,
        'exact_solution': exact_func,
        'mixed_precision': mixed_precision,
        'callbacks': callbacks,
        'model_concat_flag': models_concat_flag
    }
