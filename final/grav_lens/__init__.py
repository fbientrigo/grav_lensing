# init file
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Solo mostrar errores
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Deshabilitar oneDNN