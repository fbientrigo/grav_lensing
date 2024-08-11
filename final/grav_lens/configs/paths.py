"""
Se encarga del managing de los paths para facilitar la modelo
"""
# configs/paths.py
import os
HOME = os.getcwd()

# default values
MAX_FILES = 100
DATA_INDEX = '1'

def get_data_directory(data_index=DATA_INDEX, home=HOME):
    data_folder = os.path.join(home, 'data', data_index)
    print('Using data folder:', data_folder)
    x_data = os.path.join(data_folder,'EPSILON')
    y_data = os.path.join(data_folder,'KAPPA')
    return x_data, y_data


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

    