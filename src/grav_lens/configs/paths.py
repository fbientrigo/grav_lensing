"""
Se encarga del managing de los paths para facilitar la modelo
"""
# configs/paths.py
import os
HOME = os.getcwd()

# default values
MAX_FILES = 100
DATA_INDEX = '1'

def get_data_directory(data_index=DATA_INDEX, home=HOME, max_attempts=10):
    """
    Busca un directorio de datos específico, intentando diferentes índices numéricos si el directorio
    no existe.

    Esta función busca una carpeta de datos en el directorio especificado por 'home' y 'data_index'.
    Si no encuentra la carpeta con el 'data_index' proporcionado, intenta con otros números hasta 
    un máximo de 'max_attempts'.

    Parameters:
        data_index (str, optional): El índice de datos inicial a buscar. Por defecto es DATA_INDEX.
        home (str, optional): El directorio principal donde se encuentran los datos. Por defecto es HOME.
        max_attempts (int, optional): El número máximo de intentos para encontrar un directorio válido. Por defecto es 10.

    Returns:
        tuple: Rutas a los directorios de los datos 'EPSILON' y 'KAPPA'.

    Raises:
        FileNotFoundError: Si no se encuentra un directorio de datos válido después de 'max_attempts' intentos.

    Examples:
        >>> x_data, y_data = get_data_directory(data_index='1')
    """
    for attempt in range(int(data_index), int(data_index) + max_attempts):
        data_folder = os.path.join(home, str(attempt))
        if os.path.exists(data_folder):
            print('Using data folder:', data_folder)
            x_data = os.path.join(data_folder, 'EPSILON')
            y_data = os.path.join(data_folder, 'KAPPA')
            return x_data, y_data

    raise FileNotFoundError(f"No se encontró un directorio de datos válido después de {max_attempts} intentos a partir de {data_index}.")



def list_files_from_directory(directory, max_files=MAX_FILES):
    """
    Lista los primeros `max_files` archivos en el directorio `directory`.
    Los nombres de los archivos se almacenan en la lista global `file_names`.
    """
    file_names = []
    
    # Obtener los archivos en el directorio
    
    for root, _, files in os.walk(directory):
        # Limitar a los primeros `max_files` archivos
        for file in files[:max_files]:
            file_names.append(os.path.join(directory, file))
        break  # Salir después de la primera iteración para no recorrer subdirectorios
    
    return file_names


def get_datasets_paths_from_index(data_index=DATA_INDEX, max_files=MAX_FILES, home=HOME):
    """
    Obtiene dos valores, uno es la lista de nombres correspondientes a X, las features
    y el otro es a Y, la distribucion

    """
    x_data, y_data = get_data_directory(data_index, home)

    X = list_files_from_directory(x_data, max_files)
    Y = list_files_from_directory(y_data, max_files)

    return X, Y


# Predicción
def get_testing_paths(testing_dir, max_files=-1):
    # 1. Obtener el directorio para "testing/EPSILON"
    
    # 2. Usar `list_files_from_directory` para obtener rutas de archivos
    file_paths = list_files_from_directory(testing_dir, max_files=max_files)
    
    # 3. Devolver un generador con las rutas de archivos y los nombres base
    for file_path in file_paths:
        yield file_path, os.path.basename(file_path)








# Bloque principal para ejecutar el script
if __name__ == "__main__":
    x_data, y_data = get_data_directory(DATA_INDEX)
    print('x_data:', x_data)
    file_names = list_files_from_directory(x_data, -1)
    
    # Imprimir la lista de archivos para verificar el funcionamiento
    print(f"Primeros {len(file_names)} archivos en {x_data}:")
    for file in file_names[:10]:
        print(file)

    # Probar obtener todos los paths
    X, Y = get_datasets_paths_from_index(DATA_INDEX, 10)
    for x,y in zip(X,Y):
        print(x,'--',y)

    