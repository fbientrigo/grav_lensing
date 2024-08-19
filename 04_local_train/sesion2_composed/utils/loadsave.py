import json
import datetime
import numpy as np

def load_hyperparameters(counter):
    with open(f'hyperparameters_{counter}.json', 'r') as f:
        hyperparameters = json.load(f)
    return hyperparameters


def load_model_with_hyperparameters(counter, create_model):
    # Cargar los hiperparámetros
    hyperparameters = load_hyperparameters(counter)
    learning_rate = hyperparameters['learning_rate']
    activation = hyperparameters['activation']

    # Recrear el modelo con los hiperparámetros cargados
    model = create_model(learning_rate, activation)
    
    # Cargar los pesos del modelo
    model.load_weights(f'best_model_{counter}.weights.h5')

    return model


def save_hyperparameters(learning_rate, in_activation, h_activation, out_activation, 
                         h_kernel_size, hidden_filters, out_kernel_size, weight_kl, 
                         beta_1, beta_2, epsilon, amsgrad, decay_steps, decay_rate, counter):
    """
    Guarda los hiperparámetros en un archivo JSON.

    Args:
        learning_rate (float): Tasa de aprendizaje para el optimizador.
        in_activation (str): Función de activación para la capa de entrada.
        h_activation (str): Función de activación para las capas ocultas.
        out_activation (str): Función de activación para la capa de salida.
        h_kernel_size (int): Tamaño del kernel para las capas ocultas.
        hidden_filters (int): Número de filtros en las capas ocultas.
        out_kernel_size (int): Tamaño del kernel para la capa de salida.
        weight_kl (float): Peso para KL Divergence en la pérdida combinada.
        beta_1 (float): Parámetro beta_1 para el optimizador Adam.
        beta_2 (float): Parámetro beta_2 para el optimizador Adam.
        epsilon (float): Parámetro epsilon para el optimizador Adam.
        amsgrad (bool): Indica si se debe usar AMSGrad en el optimizador Adam.
        decay_steps (int): Número de pasos para el decaimiento del learning rate.
        decay_rate (float): Tasa de decaimiento para el learning rate.
        counter (int): Identificador para el archivo de guardado.
    """
    def serialize_value(value):
        """Convierte valores a tipos estándar de Python para asegurar compatibilidad con JSON."""
        if isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        else:
            return value

    hyperparameters = {
        'learning_rate': serialize_value(learning_rate),
        'in_activation': in_activation,
        'h_activation': h_activation,
        'out_activation': out_activation,
        'h_kernel_size': serialize_value(h_kernel_size),
        'hidden_filters': serialize_value(hidden_filters),
        'out_kernel_size': serialize_value(out_kernel_size),
        'weight_kl': serialize_value(weight_kl),
        'beta_1': serialize_value(beta_1),
        'beta_2': serialize_value(beta_2),
        'epsilon': serialize_value(epsilon),
        'amsgrad': amsgrad,
        'decay_steps': serialize_value(decay_steps),
        'decay_rate': serialize_value(decay_rate)
    }
    
    with open(f'hyperparameters_{counter}.json', 'w') as f:
        json.dump(hyperparameters, f, indent=4)



# def log_dir_name(learning_rate, in_activation, h_activation, out_activation, 
#                  h_kernel_size, hidden_filters, out_kernel_size, weight_kl, 
#                  beta_1, beta_2, epsilon, amsgrad, decay_steps, decay_rate):
#     """
#     Helper function para generar el nombre del directorio de TensorBoard.
#     """
#     s = "./logs/lr_{0:.0e}_inact_{1}_hact_{2}_outact_{3}_hks_{4}_hf_{5}_oks_{6}_wkl_{7}_beta1_{8}_beta2_{9}_eps_{10}_amsgrad_{11}_decaysteps_{12}_decayrate_{13}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
#     log_dir = s.format(learning_rate, 
#                        in_activation,
#                        h_activation,
#                        out_activation,
#                        h_kernel_size,
#                        hidden_filters,
#                        out_kernel_size,
#                        weight_kl,
#                        beta_1,
#                        beta_2,
#                        epsilon,
#                        amsgrad,
#                        decay_steps,
#                        decay_rate)

#     return log_dir