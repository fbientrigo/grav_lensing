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
            Categorical(categories=['relu', 'sigmoid', 'tanh'], name='in_activation'),
            Categorical(categories=['relu', 'sigmoid', 'tanh'], name='h_activation'),
            Categorical(categories=['relu', 'sigmoid', 'tanh'], name='out_activation'),
            Integer(low=2, high=7, name='h_kernel_size'),
            Integer(low=16, high=128, name='hidden_filters'),
            Integer(low=2, high=7, name='out_kernel_size'),
            Real(low=0.01, high=2.0, prior='log-uniform', name='weight_kl'),
            Real(low=0.0, high=0.99, name='beta_1'),
            Real(low=0.0, high=0.999, name='beta_2'),
            Real(low=1e-8, high=1e-4, prior='log-uniform', name='epsilon'),
            Categorical(categories=[True, False], name='amsgrad'),
            Integer(low=1000, high=50000, name='decay_steps'),
            Real(low=0.8, high=0.99, name='decay_rate'),
            Integer(low=5, high=25, name='epochs')
        ]
    
default_parameters = [1e-4, 'sigmoid', 'sigmoid', 'sigmoid', 3, 64, 3, 
    0.1, 0.9, 0.99, 1e-7, False, 10000, 0.96, 10]



# TODO 
# modificar para correr una vez ahora que no es parte de la clase

# def run_test_optimize(self, verbose_train=True, verbose_val=False):
#     """
#     Ejecuta una prueba rápida de optimización utilizando los parámetros predeterminados,
#     y luego restaura `best_loss` y `counter` a sus valores originales.

#     Args:
#         verbose_train (bool): Indica si se debe mostrar el progreso del entrenamiento.
#         verbose_val (bool): Indica si se debe mostrar el progreso de la evaluación.

#     Returns:
#         None
#     """
#     # Guardar el estado original de best_loss y counter
#     original_best_loss = self.best_loss
#     original_counter = self.counter

#     try:
#         start_time = time.time()
#         self.F_objective(*self.default_parameters)
#         end_time = time.time()
#         execution_time_minutes = (end_time - start_time) / 60
#         print(f"Execution time: {execution_time_minutes:.2f} minutes")
#     finally:
#         # Restaurar best_loss y counter a sus valores originales
#         self.best_loss = original_best_loss
#         self.counter = original_counter