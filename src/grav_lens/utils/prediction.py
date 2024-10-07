from os import makedirs
from os.path import join
import numpy as np

def save_predictions(prediction_batch, names_batch, home):
    """
    La función que toma la estructura del dataset predictivo para así almacenar las predicciones de un batch
    """
    # 1. Definir la carpeta de resultados
    results_dir = join(home, "RESULTS")
    makedirs(results_dir, exist_ok=True)
    
    # 2. Iterar sobre las predicciones y los nombres
    for prediction, name in zip(prediction_batch, names_batch):
        # 3. Generar la ruta completa de archivo para guardar la predicción
        result_path = join(results_dir, name.numpy().decode('utf-8'))
        try:
            np.save(result_path, prediction.astype(np.float16))

        except Exception as e:
            print(f"Error al guardar {result_path}: {e}")
        # 4. Guardar la predicción en un archivo .npy

