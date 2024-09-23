
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler



# -.-------- filtering ---------

def apply_fourier_transform(image):
    """
    Aplica la Transformada de Fourier bidimensional a una imagen.
    
    Parámetros:
        image (numpy array): Imagen en formato numpy array (2D).
        
    Retorna:
        f_transform_shifted (numpy array): Transformada de Fourier desplazada con fftshift.
    """
    f_transform = np.fft.fft2(image)
    return np.fft.fftshift(f_transform)  # Mover las frecuencias bajas al centro


def inverse_fourier_transform(f_transform_shifted):
    """
    Aplica la Transformada Inversa de Fourier a una imagen.
    
    Parámetros:
        f_transform_shifted (numpy array): Transformada de Fourier desplazada.
        
    Retorna:
        image_reconstructed (numpy array): Imagen reconstruida a partir de la transformada de Fourier.
    """
    f_transform = np.fft.ifftshift(f_transform_shifted)  # Deshacer el shift
    return np.fft.ifft2(f_transform).real  # Obtener la imagen real reconstruida




def apply_lowpass_filter(f_transform, cutoff):
    """
    Aplica un filtro pasa bajas (low-pass) a la Transformada de Fourier.
    
    Parámetros:
        f_transform (numpy array): Transformada de Fourier desplazada.
        cutoff (float): Frecuencia de corte para el filtro (entre 0 y 1, relativa al tamaño de la imagen).
        
    Retorna:
        lowpass_filtered (numpy array): Transformada de Fourier filtrada con pasa bajas.
    """
    rows, cols = f_transform.shape
    crow, ccol = rows // 2, cols // 2  # Centro de la imagen

    # Crear una máscara pasa bajas (con 1 en el centro y 0 en los bordes)
    mask = np.zeros((rows, cols), dtype=np.float32)
    radius = cutoff * min(crow, ccol)  # Radio de corte basado en la frecuencia relativa
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) <= radius:
                mask[i, j] = 1
    
    # Aplicar la máscara a la Transformada de Fourier
    lowpass_filtered = f_transform * mask
    return lowpass_filtered

def apply_highpass_filter(f_transform, cutoff):
    """
    Aplica un filtro pasa altas (high-pass) a la Transformada de Fourier.
    
    Parámetros:
        f_transform (numpy array): Transformada de Fourier desplazada.
        cutoff (float): Frecuencia de corte para el filtro (entre 0 y 1, relativa al tamaño de la imagen).
        
    Retorna:
        highpass_filtered (numpy array): Transformada de Fourier filtrada con pasa altas.
    """
    rows, cols = f_transform.shape
    crow, ccol = rows // 2, cols // 2  # Centro de la imagen

    # Crear una máscara pasa altas (con 0 en el centro y 1 en los bordes)
    mask = np.ones((rows, cols), dtype=np.float32)
    radius = cutoff * min(crow, ccol)  # Radio de corte basado en la frecuencia relativa
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) <= radius:
                mask[i, j] = 0
    
    # Aplicar la máscara a la Transformada de Fourier
    highpass_filtered = f_transform * mask
    return highpass_filtered

def process_image_with_filters(image, cutoff=0.1):
    """
    Procesa una imagen para separar las componentes de baja y alta frecuencia usando filtros pasa bajas y pasa altas.
    
    Parámetros:
        image (numpy array): Imagen en formato numpy array (2D).
        cutoff (float): Frecuencia de corte para los filtros (entre 0 y 1, relativa al tamaño de la imagen).
        
    Retorna:
        low_freq_image (numpy array): Imagen de baja frecuencia.
        high_freq_image (numpy array): Imagen de alta frecuencia.
    """
    # Aplicar la Transformada de Fourier con fftshift
    f_transform_shifted = apply_fourier_transform(image)
    
    # Aplicar filtro pasa bajas
    low_freq_transform = apply_lowpass_filter(f_transform_shifted, cutoff)
    
    # Aplicar filtro pasa altas
    high_freq_transform = apply_highpass_filter(f_transform_shifted, cutoff)
    
    # Reconstruir las imágenes de baja y alta frecuencia
    low_freq_image = inverse_fourier_transform(low_freq_transform)
    high_freq_image = inverse_fourier_transform(high_freq_transform)
    
    return low_freq_image, high_freq_image


def process_batch_filters(y_batch, cutoff=0.1):
    """
    Procesa un batch de imágenes para separar las componentes de baja y alta frecuencia.
    
    Parámetros:
        y_batch (numpy array): Batch de imágenes (batch_size, 128, 128, 1).
        threshold (float): Umbral para separar las frecuencias.
        
    Retorna:
        low_freq_batch (numpy array): Batch de imágenes de baja frecuencia.
        high_freq_batch (numpy array): Batch de imágenes de alta frecuencia.
    """
    batch_size = y_batch.shape[0]
    low_freq_batch = []
    high_freq_batch = []
    
    for idx in range(batch_size):
        image = y_batch[idx, :, :, 0]  # Seleccionar la imagen del batch
        low_freq_image, high_freq_image = process_image_with_filters(image, cutoff=0.1)
        
        low_freq_batch.append(low_freq_image)
        high_freq_batch.append(high_freq_image)
    
    # Convertir a formato numpy arrays con la misma forma que el batch original
    low_freq_batch = np.expand_dims(np.array(low_freq_batch), axis=-1)
    high_freq_batch = np.expand_dims(np.array(high_freq_batch), axis=-1)
    
    return low_freq_batch, high_freq_batch
# -------------------------------------------------------------------------

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
        return tf.convert_to_tensor(y_batch_scaled, dtype=tf.float32)  # Convertir de vuelta a tensor

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

