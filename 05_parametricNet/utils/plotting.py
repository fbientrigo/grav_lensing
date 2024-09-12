import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def plot_batch_list(list_of_batches, batch_size=4):
    """
    Dibuja las imágenes de varios batches, organizadas por fila. El número de columnas se ajusta
    automáticamente al número de batches en la lista.

    Parámetros:
        list_of_batches (list of numpy arrays): Lista de batches de imágenes. Cada batch debe tener forma (batch_size, altura, ancho, canales).
        batch_size (int): Número de imágenes a mostrar por batch (default es 4).
        
    Ejemplo de uso:
        list_of_batches = [low_freq_batch, high_freq_batch, reconstructed_batch]
        plot_batch(list_of_batches, batch_size=4)
    """
    num_batches = len(list_of_batches)  # Número de batches a graficar (número de columnas)
    fig, axs = plt.subplots(batch_size, num_batches, figsize=(5 * num_batches, 5 * batch_size))

    # Iterar a través del batch
    for i in range(batch_size):
        for j in range(num_batches):
            # Obtener la imagen del batch correspondiente
            image = list_of_batches[j][i, :, :, 0]  # Seleccionar la imagen y eliminar el canal extra
            axs[i, j].imshow(image)
            axs[i, j].set_title(f"Batch {j+1}, Image {i+1}")
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_batch_dict(dict_batches, batch_size=4):
    """
    Dibuja las imágenes de varios batches, organizadas por fila. El número de columnas se ajusta
    automáticamente según la cantidad de batches en el diccionario.

    Parámetros:
        dict_batches (dict): Diccionario donde las claves son los nombres de los batches (str) y los valores son los batches de imágenes (numpy arrays).
                             Cada batch debe tener forma (batch_size, altura, ancho, canales).
        batch_size (int): Número de imágenes a mostrar por batch (default es 4).
        
    Ejemplo de uso:
        dict_batches = {"low_freq_batch": low_freq_batch, "high_freq_batch": high_freq_batch, "reconstructed_batch": reconstructed_batch}
        plot_batch(dict_batches, batch_size=4)
    """
    batch_names = list(dict_batches.keys())  # Obtener los nombres de los batches
    num_batches = len(batch_names)  # Número de batches a graficar (número de columnas)
    fig, axs = plt.subplots(batch_size, num_batches, figsize=(5 * num_batches, 5 * batch_size))

    # Iterar a través del batch
    for i in range(batch_size):
        for j, batch_name in enumerate(batch_names):
            # Obtener la imagen del batch correspondiente
            batch = dict_batches[batch_name]
            image = batch[i, :, :, 0]  # Seleccionar la imagen y eliminar el canal extra
            axs[i, j].imshow(image)
            axs[i, j].set_title(f"{batch_name}, Image {i+1}")
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()

