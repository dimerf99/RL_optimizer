import torch
import os
import sys
import time
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data

solver_device('gpu')

data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../PINNacle_data/burgers1d.npy"))

mu = 0.01 / np.pi


def burgers_1d_experiment(grid_res):
    exp_dict_list = []

    x_min, x_max = -1, 1
    t_max = 1

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('t', [0, t_max], grid_res)

    boundaries = Conditions()

    # # Initial conditions ###############################################################################################
    #
    # # u(x, 0) = -sin(pi * x)
    # boundaries.dirichlet({'x': [x_min, x_max], 't': 0}, value=lambda grid: -torch.sin(np.pi * grid[:, 0]))
    #
    # # Boundary conditions ##############################################################################################
    #
    # # u(x_min, t) = 0
    # boundaries.dirichlet({'x': x_min, 't': [0, t_max]}, value=0)
    #
    # # u(x_max, t) = 0
    # boundaries.dirichlet({'x': x_max, 't': [0, t_max]}, value=0)

    # u(x,0)=1e4*sin^2(x(x-1)/10)
    x = domain.variable_dict['x']
    func_bnd1 = lambda x: 10 ** 4 * torch.sin((1 / 10) * x * (x - 1)) ** 2
    boundaries.dirichlet({'x': [0, 1], 't': 0}, value=func_bnd1)

    # du/dx (x,0) = 1e3*sin^2(x(x-1)/10)
    func_bnd2 = lambda x: 10 ** 3 * torch.sin((1 / 10) * x * (x - 1)) ** 2
    bop2 = {
        'du/dt':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            }
    }
    boundaries.operator({'x': [0, 1], 't': 0}, operator=bop2, value=func_bnd2)

    # u(0,t) = u(1,t)
    boundaries.periodic([{'x': 0, 't': [0, 1]}, {'x': 1, 't': [0, 1]}])

    # du/dt(0,t) = du/dt(1,t)
    bop4 = {
        'du/dx':
            {
                'coeff': 1,
                'du/dx': [0],
                'pow': 1,
                'var': 0
            }
    }
    boundaries.periodic([{'x': 0, 't': [0, 1]}, {'x': 1, 't': [0, 1]}], operator=bop4)

    equation = Equation()

    # Operator: u_t + u * u_x - mu * u_xx = 0

    burgers_eq = {
        'du/dt**1':
            {
                'coeff': 1.,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            },
        '+u*du/dx':
            {
                'coeff': 1,
                'u*du/dx': [[None], [0]],
                'pow': [1, 1],
                'var': [0, 0]
            },
        '-mu*d2u/dx2':
            {
                'coeff': -mu,
                'd2u/dx2': [0, 0],
                'pow': 1,
                'var': 0
            }
    }

    equation.add(burgers_eq)

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
    save_model = False
    callbacks = [cb_cache, cb_es]
    models_concat_flag = False

    return {
        'domain': domain,
        'conditions': boundaries,
        'equation': equation,
        'net': net,
        'exact_solution': exact_func
    }
