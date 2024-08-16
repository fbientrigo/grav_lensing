from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver
# from tensorflow.keras.callbacks import TensorBoard

from .model import create_model
from .loadsave import load_model_with_hyperparameters, load_hyperparameters, save_hyperparameters

from tensorflow.keras import backend as K
import time


# Modificables ----------------------------------------------------------------------------
# Definir las dimensiones de los hiperparámetros para la optimización
dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
dim_in_activation = Categorical(categories=['relu', 'sigmoid', 'tanh'], name='in_activation')
dim_h_activation = Categorical(categories=['relu', 'sigmoid', 'tanh'], name='h_activation')
dim_out_activation = Categorical(categories=['relu', 'sigmoid', 'tanh'], name='out_activation')

dim_h_kernel_size = Integer(low=2, high=7, name='h_kernel_size')
dim_hidden_filters = Integer(low=16, high=128, name='hidden_filters')
dim_out_kernel_size = Integer(low=2, high=7, name='out_kernel_size')

dim_weight_kl = Real(low=0.01, high=2.0, prior='log-uniform', name='weight_kl')
dim_beta_1 = Real(low=0.0, high=0.99, name='beta_1')
dim_beta_2 = Real(low=0.0, high=0.999, name='beta_2')

dim_epsilon = Real(low=1e-8, high=1e-4, prior='log-uniform', name='epsilon')
dim_amsgrad = Categorical(categories=[True, False], name='amsgrad')
dim_decay_steps = Integer(low=1000, high=50000, name='decay_steps')

dim_decay_rate = Real(low=0.8, high=0.99, name='decay_rate')
dim_epochs = Integer(low=5, high=25, name='epochs')


# Valores predeterminados para los hiperparámetros
default_parameters = [1e-4, 'sigmoid', 
                      'sigmoid', 'sigmoid', 
                      3, 64, 3, 
                      0.1, 0.9, 0.99, 
                      1e-7, False, 10000, 
                      0.96, 10]






# ----------------------------
# Lista de todas las dimensiones
dimensions = [
    dim_learning_rate, dim_in_activation, 
    dim_h_activation, dim_out_activation,
    dim_h_kernel_size, dim_hidden_filters, dim_out_kernel_size,
    dim_weight_kl, dim_beta_1, dim_beta_2,
    dim_epsilon, dim_amsgrad, dim_decay_steps, 
    dim_decay_rate, dim_epochs
]


# Función objetivo
def create_F_objective(train_dataset, val_dataset, verbose_train=True, verbose_val=False):
    @use_named_args(dimensions=dimensions)
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

        # log_dir = log_dir_name(
        #     learning_rate, in_activation, 
        #     h_activation, out_activation, 
        #     h_kernel_size, hidden_filters, 
        #     out_kernel_size, weight_kl, 
        #     beta_1, beta_2, epsilon, amsgrad, 
        #     decay_steps, decay_rate
        # )
        # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)

        # Entrenar el modelo
        model.fit(train_dataset, epochs=epochs, verbose=verbose_train)
            #, callbacks=[tensorboard_callback])

        loss = model.evaluate(val_dataset, verbose=verbose_val)

        print(f"\nLoss: {loss:.2%}\n")

        global best_loss, counter
        if loss < best_loss:
            model.save_weights(f'best_model_{counter}.weights.h5')
            save_hyperparameters(
                learning_rate, in_activation, 
                h_activation, out_activation, 
                h_kernel_size, hidden_filters, 
                out_kernel_size, weight_kl, 
                beta_1, beta_2, epsilon, 
                amsgrad, decay_steps, 
                decay_rate, counter
            )
            print(f"Model weights and hyperparameters saved with ID: {counter}")
            # print(f"TensorBoard logs directory: {log_dir}")
            counter += 1
            best_loss = loss

        K.clear_session()
        return loss
    return F_objective

def run_test_optimize(train_dataset, val_dataset, n_calls=17, verbose_train=True, verbose_val=False):
    checkpoint_saver = CheckpointSaver("checkpoint.pkl", compress=9)

    # Ejecutar la optimización
    start_time = time.time()
    res = gp_minimize(
        func=create_F_objective(train_dataset, val_dataset, verbose_train, verbose_val),
        dimensions=dimensions,
        acq_func='EI', 
        n_calls=n_calls,
        x0=default_parameters,
        callback=[checkpoint_saver])
    end_time = time.time()

    execution_time_minutes = (end_time - start_time) / 60
    print(f"Execution time: {execution_time_minutes:.2f} minutes")
    return res
