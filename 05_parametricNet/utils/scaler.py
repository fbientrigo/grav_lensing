import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

class CustomScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_scaler(self, dataset):
        """
        Función para ajustar el StandardScaler a los datos de entrenamiento.
        
        Parámetros:
        - dataset: Un iterable o generador de datos que contiene las imágenes en lotes (batch_size, height, width, 1)
        """
        y_data = []
        for X_batch, y_batch in dataset:
            y_batch = y_batch.numpy().reshape(-1)  # Aplanar el batch de imágenes
            y_data.extend(y_batch)

        y_data = np.array(y_data).reshape(-1, 1)
        self.scaler.fit(y_data)  # Ajustar el scaler a los datos

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