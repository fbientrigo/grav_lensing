import pickle
import pkg_resources
import numpy as np

from tensorflow import convert_to_tensor, float32

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import IncrementalPCA 

# ------ Scaler  ---------------
class CustomMinMaxScaler:
    def __init__(self, feature_range=(0, 1), clip=True):
        """
        Inicializa el MinMaxScaler con el rango de características deseado.
        
        Parámetros:
        - feature_range: El rango en el que los datos serán escalados. Por defecto es (0, 1).
        """
        self.scaler = MinMaxScaler(feature_range=feature_range, clip=clip)

    def fit_scaler(self, dataset):
        """
        Función para ajustar el MinMaxScaler a los datos de entrenamiento.
        
        Parámetros:
        - dataset: Un iterable o generador de datos que contiene las imágenes en lotes (batch_size, height, width, 1)
        """

        for X_batch, y_batch in dataset:
            y_batch = y_batch.numpy().reshape(-1)  # Aplanar el batch de imágenes
            y_data = np.array(y_batch).reshape(-1, 1)
            self.scaler.partial_fit(y_data)  # Ajustar el scaler a los datos



    def transform(self, y_batch):
        """
        Función para normalizar un batch de datos usando el scaler ajustado.
        
        Parámetros:
        - y_batch: Un batch de imágenes en forma de tensor (batch_size, height, width, 1)
        
        Retorna:
        - El batch normalizado.
        """
        y_batch = y_batch.numpy()  # Convertir a numpy
        y_batch_flat = y_batch.reshape(-1, 1)  # Aplanar
        y_batch_scaled = self.scaler.transform(y_batch_flat)  # Escalar
        y_batch_scaled = y_batch_scaled.reshape(y_batch.shape)  # Volver a la forma original
        return convert_to_tensor(y_batch_scaled, dtype=float32)  # Convertir de vuelta a tensor

    def inverse_transform(self, y_batch_scaled):
        """
        Función para desescalar un batch de datos usando el scaler ajustado.
        
        Parámetros:
        - y_batch_scaled: Un batch normalizado de imágenes (batch_size, height, width, 1)
        
        Retorna:
        - El batch desescalado a los valores originales.
        """
        y_batch_flat = y_batch_scaled.reshape(-1, 1)  # Aplanar
        y_batch_original = self.scaler.inverse_transform(y_batch_flat)  # Desescalar
        y_batch_original = y_batch_original.reshape(y_batch_scaled.shape)  # Volver a la forma original
        return y_batch_original



def load_minmaxscaler():
    """
    Carga el objeto MinMaxScaler desde el archivo minmaxscaler.pkl empaquetado en la librería.

    Este método utiliza `pkg_resources` para localizar y cargar el archivo preentrenado 
    `minmaxscaler.pkl` que está incluido dentro del paquete `grav_lens.models`.

    Returns:
        sklearn.preprocessing.MinMaxScaler: Objeto MinMaxScaler cargado desde el archivo pickle.

    Raises:
        FileNotFoundError: Si el archivo minmaxscaler.pkl no se encuentra en el paquete.
        pickle.UnpicklingError: Si ocurre un error durante la deserialización del archivo pickle.

    Example:
        >>> minmaxscaler = load_minmaxscaler()
        >>> print(minmaxscaler)
    """
    scaler_path = pkg_resources.resource_filename('grav_lens.models', 'minmaxscaler.pkl')
    with open(scaler_path, 'rb') as f:
        minmaxscaler = pickle.load(f)
    return minmaxscaler

def load_ipca_low():
    """
    Carga el objeto Incremental PCA (IPCA) desde el archivo ipca_low.pkl empaquetado en la librería.

    Este método utiliza `pkg_resources` para localizar y cargar el archivo preentrenado 
    `ipca_low.pkl` que está incluido dentro del paquete `grav_lens.models`.

    Returns:
        sklearn.decomposition.IncrementalPCA: Objeto IPCA cargado desde el archivo pickle.

    Raises:
        FileNotFoundError: Si el archivo ipca_low.pkl no se encuentra en el paquete.
        pickle.UnpicklingError: Si ocurre un error durante la deserialización del archivo pickle.

    Example:
        >>> ipca_low = load_ipca_low()
        >>> print(ipca_low)
    """
    pca_path = pkg_resources.resource_filename('grav_lens.models', 'ipca_low.pkl')
    with open(pca_path, 'rb') as f:
        ipca_low = pickle.load(f)
    return ipca_low
