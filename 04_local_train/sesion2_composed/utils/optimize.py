from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver
import time

from .model import create_model
from .loadsave import save_hyperparameters
from tensorflow.keras import backend as K
from functools import partial


dimensions = [
            Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate'),
            Integer(low=2, high=7, name='h_kernel_size'),
            Integer(low=16, high=128, name='hidden_filters'),
            Integer(low=2, high=7, name='out_kernel_size'),
            Real(low=0.0, high=0.99, name='beta_1'),
            Real(low=0.0, high=0.999, name='beta_2'),
            Real(low=1e-8, high=1e-4, prior='log-uniform', name='epsilon'),
            Integer(low=1000, high=50000, name='decay_steps'),
            Real(low=0.8, high=0.99, name='decay_rate'),
            Integer(low=5, high=25, name='epochs')
        ]
    
default_parameters = [1e-4, 3, 64, 3, 
    0.9, 0.99, 1e-7, 10000, 0.96, 10]

