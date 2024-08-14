import os
import sys
import numpy as np

from grav_lens.configs.paths import get_data_directory, list_files_from_directory, get_datasets_paths_from_index

import tensorflow as tf

from sklearn.model_selection import train_test_split

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
            tf.TensorSpec(shape=(128, 128, 1), dtype=tf.float32)
        )
    )

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

# # Seccion encargada de conseguir training,validation y testing ------------
# def split_dataset(X_data, Y_data, val_split=0.2, test_split=0.1):
#     # Dividir los datos en entrenamiento y conjunto temporal
#     X_train, X_temp, Y_train, Y_temp = train_test_split(X_data, Y_data, test_size=val_split + test_split)
    
#     # Dividir los datos temporales en validación y prueba
#     X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=test_split / (val_split + test_split))
    
#     # Convertir a dataset de TensorFlow
#     train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
#     val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
#     test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
#     return train_dataset, val_dataset, test_dataset

def prepare_dataset(dataset, batch_size=32, shuffle_buffer=1000):
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)  # Mezclar datos
    dataset = dataset.batch(batch_size)  # Batching
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # Prefetch para optimización
    return dataset

def get_datasets(data_index=4, max_files=100, home='..', batch_size=32, val_split=0.2, test_split=0.1):
    """
    Obtiene y divide los datasets en entrenamiento, validación y prueba,
    y los prepara para el entrenamiento con el tamaño de batch especificado.

    Parameters:
        data_index (str, optional): Índice o identificador para el dataset a cargar. Por defecto es '4'.
        max_files (int, optional): Número máximo de archivos a cargar. Por defecto es 100.
        home (str, optional): Directorio principal para el almacenamiento del dataset. Por defecto es '..'.
        batch_size (int, optional): Tamaño del batch para el dataset. Por defecto es 32.
        val_split (float, optional): Proporción de datos para el conjunto de validación. Por defecto es 0.2.
        test_split (float, optional): Proporción de datos para el conjunto de prueba. Por defecto es 0.1.

    Returns:
        tuple: Tres datasets de TensorFlow (train_dataset, val_dataset, test_dataset).
    """
    # Obtener rutas de archivos
    X_paths, Y_paths = get_datasets_paths_from_index(data_index=str(data_index), max_files=max_files, home=home)
    
    # Mezclar rutas para aleatorizar
    combined_paths = list(zip(X_paths, Y_paths))
    np.random.shuffle(combined_paths)
    X_paths, Y_paths = zip(*combined_paths)
    
    # Calcular índices de división
    total_files = len(X_paths)
    val_size = int(total_files * val_split)
    test_size = int(total_files * test_split)
    
    # Dividir rutas en entrenamiento, validación y prueba
    X_train_paths, X_val_paths, X_test_paths = X_paths[:-val_size-test_size], X_paths[-val_size-test_size:-test_size], X_paths[-test_size:]
    Y_train_paths, Y_val_paths, Y_test_paths = Y_paths[:-val_size-test_size], Y_paths[-val_size-test_size:-test_size], Y_paths[-test_size:]
    
    # Crear datasets para cada conjunto
    train_dataset = create_tf_dataset(X_train_paths, Y_train_paths)
    val_dataset = create_tf_dataset(X_val_paths, Y_val_paths)
    test_dataset = create_tf_dataset(X_test_paths, Y_test_paths)
    
    # Preparar cada dataset
    train_dataset = prepare_dataset(train_dataset, batch_size)
    val_dataset = prepare_dataset(val_dataset, batch_size)
    test_dataset = prepare_dataset(test_dataset, batch_size)
    
    return train_dataset, val_dataset, test_dataset

# --------------------



if __name__ == "__main__":
    dataset = load_tf_dataset(data_index=DATA_INDEX, max_files=MAX_FILES)

    # Imprimir información básica del dataset para verificar
    for X, Y in dataset.take(5):  # Solo mostrar los primeros 5 elementos
        print("X:", X.numpy())
        print("Y:", Y.numpy())
