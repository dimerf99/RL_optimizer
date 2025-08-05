import torch
import numpy as np

from tedeous.data import Domain, Conditions, Equation
from tedeous.device import solver_device
from tedeous.callbacks.early_stopping import EarlyStopping
from tedeous.callbacks.cache import Cache

solver_device('gpu')


def exact_func(grid):
    x, t = grid[:, 0], grid[:, 1]
    sln = torch.cos(2 * np.pi * t) * torch.sin(np.pi * x)
    return sln


def get_problem_config(grid_res: int):
    domain = Domain()
    domain.variable('x', [0, 1], grid_res)
    domain.variable('t', [0, 1], grid_res)

    boundaries = Conditions()
    boundaries.dirichlet({'x': [0, 1], 't': 0}, value=exact_func)
    bop = {
        'du/dt':
            {
                'coeff': 1,
                'term': [1],
                'pow': 1,
                'var': 0
            }
    }
    boundaries.operator({'x': [0, 1], 't': 0}, operator=bop, value=0)
    boundaries.dirichlet({'x': 0, 't': [0, 1]}, value=exact_func)
    boundaries.dirichlet({'x': 1, 't': [0, 1]}, value=exact_func)

    equation = Equation()
    wave_eq = {
        'd2u/dt2**1': {'coeff': 1, 'd2u/dt2': [1, 1], 'pow': 1},
        '-C*d2u/dx2**1': {'coeff': -4, 'd2u/dx2': [0, 0], 'pow': 1}
    }
    equation.add(wave_eq)

    neurons = 32

    net = torch.nn.Sequential(
        torch.nn.Linear(2, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, 1)
    )

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    model_layers = [2, neurons, neurons, 1]
    grid = domain.build('NN').to('cuda')
    grid_test = torch.cartesian_prod(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
    u_exact_test = exact_func(grid_test).reshape(-1)
    loss_surface_equation_params = [u_exact_test, grid_test, grid, domain, equation, boundaries, model_layers]

    cb_cache = Cache(cache_verbose=True, model_randomize_parameter=1e-6)
    cb_es = EarlyStopping(eps=1e-6,
                          loss_window=100,
                          no_improvement_patience=100,
                          patience=20,
                          randomize_parameter=1e-4,
                          info_string_every=1)

    return {
        'exact_solution': exact_func,
        'domain': domain,
        'conditions': boundaries,
        'equation': equation,
        'net': net,
        'loss_surface_equation_params': loss_surface_equation_params,
        'compile_params': {
            'mode': 'autograd',
            'lambda_operator': 1,
            'lambda_bound': 100
        },
        'train_params': {
            'mixed_precision': False,
            'save_model': False,
            'callbacks': [cb_cache, cb_es],
            'models_concat_flag': False,
        }
    }
