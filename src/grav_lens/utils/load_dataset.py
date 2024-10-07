import os
import sys
import numpy as np

from grav_lens.configs.paths import get_data_directory, list_files_from_directory, get_datasets_paths_from_index, get_testing_paths

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

def load_npy_file(file_path):
    """
    Function to load a .npy file and preprocess it.
    """
    data = np.load(file_path.decode('utf-8'))
    
    # Example processing: reshape or transpose if needed
    if data.ndim == 3:  # Assuming shape is (channels, height, width)
        data = np.transpose(data, (1, 2, 0))  # Convert to (height, width, channels)
    return data.astype(np.float32)


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

def file_path_generator(X_paths, Y_paths):
    """
    Generator function that yields file paths incrementally.
    """
    for X_path, Y_path in zip(X_paths, Y_paths):
        yield X_path, Y_path

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

    dataset = tf.data.Dataset.from_generator(
        lambda: file_path_generator(X_paths, Y_paths),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )
    )
    

    dataset = dataset.map(
        lambda X_path, Y_path: (
            tf.numpy_function(load_npy_file, [X_path], tf.float32),
            tf.numpy_function(load_npy_file, [Y_path], tf.float32)
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    

    dataset = dataset.map(
        lambda X, Y: (
            tf.ensure_shape(X, [128, 128, 3]),
            tf.ensure_shape(tf.expand_dims(Y, axis=-1), [128, 128, 1])
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
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

# ---------------------------

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

def prepare_dataset(dataset, batch_size=32, shuffle_buffer=1000, drop_remainder=False, shuffle=True):
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)  # Mezclar datos
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)  # Batching

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # Prefetch para optimización
    return dataset


def warnings_size_dataset(total_files, train_size, val_size, test_size, batch_size):
    """
    Comprueba que existan suficientes muestras de cada uno,
    al crear batchs pueden quedar muestras vacias lo que provoca que no entrene con ese batch
    por tanto se ha optado por eliminar los batchs incompletos.

    Sin embargo en el caso de usar pocos datos y un batch_size significativo, pueden quedar batch vacios
    """
    # Validar que haya suficientes datos para al menos un batch en validación y prueba
    if val_size//batch_size == 0:
        warnings.warn(f"El conjunto de validación tiene solo {val_size} muestras, lo que no es suficiente para formar un batch completo. Considere ajustar 'val_split' o 'batch_size'.")
        val_size = batch_size  # Ajusta el tamaño de validación para que haya al menos un batch completo.
    
    if test_size//batch_size == 0:
        warnings.warn(f"El conjunto de prueba tiene solo {test_size} muestras, lo que no es suficiente para formar un batch completo. Considere ajustar 'test_split' o 'batch_size'.")
        test_size = batch_size  # Ajusta el tamaño de prueba para que haya al menos un batch completo.
    
    # Asegurarse de que los datos de entrenamiento no queden vacíos
    train_size = total_files - val_size - test_size
    if train_size//batch_size == 0:
        raise ValueError(f"No hay suficientes datos para entrenamiento. Después de la división, el conjunto de entrenamiento tiene solo {train_size} muestras. Ajuste las proporciones o el tamaño del batch.")
    
    # Dividir 



def get_datasets(data_index=4, max_files=100, home='..', batch_size=32, 
    val_split=0.2, test_split=0.1):
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
    X_paths, Y_paths = zip(*combined_paths)
    
    # Calcular índices de división
    total_files = len(X_paths)
    val_size = int(total_files * val_split)
    test_size = int(total_files * test_split)

    #warnings_size_dataset(total_files, train_size, val_size, test_size, batch_size)
    
    # Dividir rutas en entrenamiento, validación y prueba
    X_train_paths, X_val_paths, X_test_paths = X_paths[:-val_size-test_size], X_paths[-val_size-test_size:-test_size], X_paths[-test_size:]
    Y_train_paths, Y_val_paths, Y_test_paths = Y_paths[:-val_size-test_size], Y_paths[-val_size-test_size:-test_size], Y_paths[-test_size:]
    
    # Crear datasets para cada conjunto
    train_dataset = create_tf_dataset(X_train_paths, Y_train_paths)
    val_dataset = create_tf_dataset(X_val_paths, Y_val_paths)
    test_dataset = create_tf_dataset(X_test_paths, Y_test_paths)
    
    # Preparar cada dataset
    train_dataset = prepare_dataset(train_dataset, batch_size, drop_remainder=True)
    val_dataset = prepare_dataset(val_dataset, batch_size)
    test_dataset = prepare_dataset(test_dataset, batch_size)
    
    return train_dataset, val_dataset, test_dataset

# -------------------- predicción : -------------

def load_testing_dataset(dataset_path, batch_size=32, max_files=-1):
    # 1. Obtener rutas y nombres de archivos usando `get_testing_paths`
    paths_and_names_gen = get_testing_paths(dataset_path, max_files=max_files)
    
    # 2. Crear un Dataset de TensorFlow a partir del generador
    dataset = tf.data.Dataset.from_generator(
        lambda: paths_and_names_gen,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),  # Ruta de archivo
            tf.TensorSpec(shape=(), dtype=tf.string)   # Nombre de archivo
        )
    )
    
    # 3. Cargar los archivos usando `tf.numpy_function`
    dataset = dataset.map(
        lambda path, name: (
            tf.numpy_function(load_npy_file, [path], tf.float32),
            name
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    # 4. Asegurar el formato correcto de las características (X)
    dataset = dataset.map(
        lambda X, name: (tf.ensure_shape(X, [128, 128, 3]), name),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    # 5. Preparar el dataset (batch, prefetch)
    dataset = prepare_dataset(dataset, batch_size=batch_size, shuffle=False)
    
    return dataset


# ------------











if __name__ == "__main__":
    home = os.path.join(
        "..","..","..", "data"
    )
    print(home)
    dataset = load_tf_dataset(data_index=DATA_INDEX, max_files=MAX_FILES, home=home)

    # Imprimir información básica del dataset para verificar
    for X, Y in dataset.take(5):  # Solo mostrar los primeros 5 elementos
        print("X:", X.numpy())
        print("Y:", Y.numpy())
