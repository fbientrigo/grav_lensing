import os
import sys
import numpy as np

from grav_lens.configs.paths import get_data_directory, list_files_from_directory, get_datasets_paths_from_index

import tensorflow as tf

# Eventualmente, este código de carga de datasets se moverá a utils
# from grav_lens.utils.dataset import load_dataset

"""
Objetivos del script:
- A partir de la lista de nombres de archivos (.npy) para características (features) y etiquetas (labels),
  cargar los datos en un TensorFlow Dataset.
- Realizar pruebas simples sobre el dataset cargado.
- Calcular el tamaño en memoria de los datasets cargados.
"""

# defaults values
DATA_INDEX = '1'
MAX_FILES = 1000 # -1 usa todos los ficheros

def load_npy_files(file_paths):
    """
    Carga archivos .npy a partir de una lista de rutas de archivos.
    
    Parameters:
        file_paths (list): Lista de rutas a los archivos .npy.
        
    Returns:
        list: Lista de arrays cargados desde los archivos .npy.
    """
    return [np.load(file) for file in file_paths]

def calculate_memory_size(arrays):
    """
    Calcula el tamaño total en memoria de una lista de arrays de numpy.
    
    Parameters:
        arrays (list): Lista de numpy arrays.
        
    Returns:
        int: Tamaño total en memoria en bytes.
    """
    return sum(array.nbytes for array in arrays)

def create_tf_dataset(X_paths, Y_paths):
    """
    Crea un TensorFlow Dataset a partir de listas de rutas de archivos .npy para X (features) e Y (labels).
    
    Parameters:
        X_paths (list): Lista de rutas a los archivos .npy de características.
        Y_paths (list): Lista de rutas a los archivos .npy de etiquetas.
        
    Returns:
        tf.data.Dataset: Dataset de TensorFlow para X e Y.
    """
    X_data = load_npy_files(X_paths)
    Y_data = load_npy_files(Y_paths)

    # Verificación de integridad
    assert len(X_data) == len(Y_data), "El número de muestras en X y Y debe coincidir."
    
    # Convertir listas de numpy arrays a TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_data, Y_data))
    #return dataset, X_data, Y_data # sobrecargara la memoria
    return dataset

def load_tf_dataset(data_index=DATA_INDEX, max_files=MAX_FILES):
    X_paths, Y_paths = get_datasets_paths_from_index(data_index=data_index, max_files=max_files)
    dataset = create_tf_dataset(X_paths, Y_paths)

    # if debug:
    #     print(len(X_data))
    #     X_memory_size = calculate_memory_size(X_data)
    #     Y_memory_size = calculate_memory_size(Y_data)
    #     print(f"Tamaño en memoria de X: {X_memory_size / (1024**2):.2f} MB")
    #     print(f"Tamaño en memoria de Y: {Y_memory_size / (1024**2):.2f} MB")

    return dataset

if __name__ == "__main__":
    dataset = load_tf_dataset(data_index=DATA_INDEX, max_files=MAX_FILES)

    # Imprimir información básica del dataset para verificar
    for X, Y in dataset.take(5):  # Solo mostrar los primeros 5 elementos
        print("X:", X.numpy())
        print("Y:", Y.numpy())
