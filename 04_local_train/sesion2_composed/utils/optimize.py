from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver
from .model import create_model
from .loadsave import save_hyperparameters
from tensorflow.keras import backend as K
import time

class HyperparameterOptimizer:
    def __init__(self):
        self.best_loss = float('inf')
        self.counter = 1

        # Definir las dimensiones de los hiperparámetros para la optimización
        self.dimensions = [
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

        # Valores predeterminados para los hiperparámetros
        self.default_parameters = [1e-4, 'sigmoid', 'sigmoid', 'sigmoid', 3, 64, 3, 
                                   0.1, 0.9, 0.99, 1e-7, False, 10000, 0.96, 10]

    def create_F_objective(self, train_dataset, val_dataset, verbose_train=True, verbose_val=False):
        @use_named_args(dimensions=self.dimensions)
        def F_objective(learning_rate, 
                        in_activation, h_activation, out_activation, 
                        h_kernel_size, hidden_filters, 
                        out_kernel_size, weight_kl, 
                        beta_1, beta_2, epsilon, amsgrad, 
                        decay_steps, decay_rate, epochs):
            """
            Función objetivo para la optimización de hiperparámetros.
            """
            model = create_model(learning_rate, 
                                in_activation, h_activation, 
                                out_activation, h_kernel_size, 
                                hidden_filters, out_kernel_size, 
                                weight_kl, beta_1, beta_2, 
                                epsilon, amsgrad, 
                                decay_steps, decay_rate)

            model.fit(train_dataset, epochs=epochs, verbose=verbose_train)

            loss = model.evaluate(val_dataset, verbose=verbose_val)

            print(f"\nLoss: {loss:.2%}\n")

            if loss < self.best_loss:
                model.save_weights(f'best_model_{self.counter}.weights.h5')
                save_hyperparameters(
                    learning_rate, in_activation, 
                    h_activation, out_activation, 
                    h_kernel_size, hidden_filters, 
                    out_kernel_size, weight_kl, 
                    beta_1, beta_2, epsilon, 
                    amsgrad, decay_steps, 
                    decay_rate, self.counter
                )
                print(f"Model weights and hyperparameters saved with ID: {self.counter}")
                self.counter += 1
                self.best_loss = loss

            K.clear_session()
            return loss
        return F_objective

    def run_test_optimize(self, train_dataset, val_dataset, verbose_train=True, verbose_val=False):
        """
        Ejecuta una prueba rápida de optimización utilizando los parámetros predeterminados,
        y luego restaura `best_loss` y `counter` a sus valores originales.

        Args:
            train_dataset (tf.data.Dataset): Dataset de entrenamiento.
            val_dataset (tf.data.Dataset): Dataset de validación.
            verbose_train (bool): Indica si se debe mostrar el progreso del entrenamiento.
            verbose_val (bool): Indica si se debe mostrar el progreso de la evaluación.

        Returns:
            None
        """
        # Guardar el estado original de best_loss y counter
        original_best_loss = self.best_loss
        original_counter = self.counter

        try:
            start_time = time.time()
            self.create_F_objective(train_dataset, val_dataset, verbose_train, verbose_val)(x=self.default_parameters)
            end_time = time.time()
            execution_time_minutes = (end_time - start_time) / 60
            print(f"Execution time: {execution_time_minutes:.2f} minutes")
        finally:
            # Restaurar best_loss y counter a sus valores originales
            self.best_loss = original_best_loss
            self.counter = original_counter

    def run_hyp_optimize(self, train_dataset, val_dataset, n_calls=17, verbose_train=True, verbose_val=False):
        checkpoint_saver = CheckpointSaver("checkpoint.pkl", compress=9)
        min_function = self.create_F_objective(train_dataset, val_dataset, verbose_train, verbose_val)
        start_time = time.time()
        res = gp_minimize(
            func=min_function,
            dimensions=self.dimensions,
            acq_func='EI', 
            n_calls=n_calls,
            x0=self.default_parameters,
            callback=[checkpoint_saver]
        )
        end_time = time.time()

        execution_time_minutes = (end_time - start_time) / 60
        print(f"Execution time: {execution_time_minutes:.2f} minutes")
        return res
