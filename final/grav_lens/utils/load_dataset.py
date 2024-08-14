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
HOME = '.'

def load_npy_files(file_paths):
    """
    Carga archivos .npy desde una lista de rutas de archivos.

    Esta función carga múltiples archivos .npy, lo cual es común en flujos de trabajo científicos que 
    involucran grandes conjuntos de datos.

    Parameters:
        rutas_archivos (list of str): Lista de rutas a los archivos .npy.

    Returns:
        list of numpy.ndarray: Lista de arrays cargados desde los archivos .npy.

    Raises:
        FileNotFoundError: Si alguno de los archivos en la lista no existe.
        IOError: Si ocurre un error al intentar cargar un archivo.

    Examples:
        >>> rutas_archivos = ['data1.npy', 'data2.npy']
        >>> arrays = load_npy_files(rutas_archivos)
    """
    return [np.load(file) for file in file_paths]

def calculate_memory_size(arrays):
    """
    Calcula el tamaño total en memoria de una lista de arrays de numpy.

    Esta función suma el uso de memoria de cada array en la lista.

    Parameters:
        arrays (list of numpy.ndarray): Lista de numpy arrays.

    Returns:
        int: Tamaño total en memoria en bytes.

    Examples:
        >>> arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        >>> total_size = calculate_memory_size(arrays)
    """
    return sum(array.nbytes for array in arrays)

def data_generator(X_paths, Y_paths):
    """
    Formateo necesario para que llevar los datos a la forma correcta que aceptan los algoritmos
    """
    for X_path, Y_path in zip(X_paths, Y_paths):
        X = np.load(X_path)
        Y = np.load(Y_path)

        # Verifica si X está en formato batch (4 dimensiones)
        if X.ndim == 4:
            # Reordena el eje si es necesario para el formato batch (4 dimensiones)
            X = np.transpose(X, (0, 2, 3, 1))  # Cambia (batch_size, channels, height, width) a (batch_size, height, width, channels)
        else:
            # Si X no está en formato batch (3 dimensiones), simplemente reordena para (height, width, channels)
            X = np.transpose(X, (1, 2, 0))  # Cambia (channels, height, width) a (height, width, channels)

        # Expande las dimensiones de Y para el formato (height, width, 1)
        
        Y = np.expand_dims(Y, axis=-1)  # Cambia (height, width) a (height, width, 1)
        
        yield X, Y

def create_tf_dataset(X_paths, Y_paths):
    """
    Crea un Dataset de TensorFlow a partir de listas de rutas de archivos .npy para X (características) y Y (etiquetas).

    Esta función carga los datos desde archivos .npy y construye un Dataset de TensorFlow, asegurando que 
    el número de muestras en X y Y coincida.

    Parameters:
        rutas_X (list of str): Lista de rutas a los archivos .npy que contienen las características.
        rutas_Y (list of str): Lista de rutas a los archivos .npy que contienen las etiquetas.

    Returns:
        tf.data.Dataset: Dataset de TensorFlow que contiene pares de datos X e Y.

    Raises:
        AssertionError: Si el número de muestras en X y Y no coincide.

    Examples:
        >>> rutas_X = ['X_data1.npy', 'X_data2.npy']
        >>> rutas_Y = ['Y_data1.npy', 'Y_data2.npy']
        >>> dataset = create_tf_dataset(rutas_X, rutas_Y)
    """
    X_data = load_npy_files(X_paths)
    Y_data = load_npy_files(Y_paths)

    # Verificación de integridad
    assert len(X_data) == len(Y_data), "El número de muestras en X y Y debe coincidir."
    
    # Convertir listas de numpy arrays a TensorFlow Dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(X_paths, Y_paths),
        output_signature=(
            tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(128, 128,1 ), dtype=tf.float32)
        )
    )
    #return dataset, X_data, Y_data # sobrecargara la memoria
    return dataset

def load_tf_dataset(data_index=DATA_INDEX, max_files=MAX_FILES, home=HOME):
    """
    Carga un Dataset de TensorFlow a partir de listas de rutas de archivos .npy para X (características) y Y (etiquetas).

    Esta función recupera las rutas de los archivos usando un índice de datos y crea un Dataset de TensorFlow.

    Parameters:
        data_index (str, optional): Índice o identificador para el dataset a cargar. Por defecto es DATA_INDEX.
        max_files (int, optional): Número máximo de archivos a cargar. Por defecto es MAX_FILES.
        home (str, optional): Directorio principal para el almacenamiento del dataset. Por defecto es HOME.

    Returns:
        tf.data.Dataset: Dataset de TensorFlow que contiene datos X e Y.

    Dependencias:
        get_datasets_paths_from_index
        create_tf_dataset

    Examples:
        >>> dataset = load_tf_dataset(data_index='001', max_files=10)
    """
    X_paths, Y_paths = get_datasets_paths_from_index(data_index=str(data_index), max_files=max_files, home=home)
    dataset = create_tf_dataset(X_paths, Y_paths)

    return dataset

if __name__ == "__main__":
    dataset = load_tf_dataset(data_index=DATA_INDEX, max_files=MAX_FILES)

    # Imprimir información básica del dataset para verificar
    for X, Y in dataset.take(5):  # Solo mostrar los primeros 5 elementos
        print("X:", X.numpy())
        print("Y:", Y.numpy())
