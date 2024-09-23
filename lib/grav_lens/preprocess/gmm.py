
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# ----------- Funciones Basicas de las que dependen las otras ----------------
def calculate_image_stats(image):
    """
    Calcula la media y la desviación estándar de la imagen.
    """
    mean_value = np.mean(image)
    std_value = np.std(image)
    return mean_value, std_value


def apply_threshold(image, mean_value, std_value, threshold=1.0):
    """
    Aplica un umbral a la imagen basado en la media y la desviación estándar.

    Parámetros:
        image (numpy array): Imagen en formato numpy array.
        mean_value (float): Media de los valores de la imagen.
        std_value (float): Desviación estándar de los valores de la imagen.
        threshold (float): Factor de umbral para definir el rango de valores que se mantendrán.

    Retorna:
        image_thresholded (numpy array): Imagen con valores filtrados.
    """
    lower_bound = mean_value - threshold * std_value
    upper_bound = mean_value + threshold * std_value
    
    # Filtrar la imagen utilizando el umbral
    image_thresholded_positive = np.where((image >= upper_bound), image , 0)
    image_thresholded_negative = np.where((image <= lower_bound), image, 0)
    
    return image_thresholded_positive, image_thresholded_negative




def generate_points_from_image(image, n_samples=100):
    """
    Genera una cantidad fija de puntos (x, y) a partir de una imagen, donde la cantidad de puntos es proporcional a la intensidad del pixel.
    
    Parámetros:
        image (numpy array): Imagen en formato numpy array (2D).
        n_samples (int): Número total de puntos a generar.
        
    Retorna:
        points (numpy array): Array de coordenadas (x, y) generadas a partir de los valores de los píxeles.
    """
    # Verificar si la imagen está vacía (todos los valores son cero)
    if np.sum(image) == 0:
        # Si está vacía, devolver una lista vacía de puntos
        return np.array([])
    
    # Aplanar la imagen para simplificar el muestreo
    flattened_image = image.flatten()
    
    # Normalizar la imagen para que sus valores sumen 1 (esto crea una distribución de probabilidad)
    probabilities = flattened_image / np.sum(flattened_image)
    
    # Generar los índices de los píxeles con muestreo ponderado según las intensidades
    chosen_indices = np.random.choice(len(flattened_image), size=n_samples, p=probabilities)
    
    # Convertir los índices planos en coordenadas (x, y)
    x_coords, y_coords = np.unravel_index(chosen_indices, image.shape)
    
    # Combinar las coordenadas en una lista de puntos (x, y)
    points = np.vstack((x_coords, y_coords)).T
    
    return points

def gaussian_2d(x, y, mux, muy, sigma_x, sigma_y, weight):
    """
    Genera una Gaussiana bidimensional.

    Parámetros:
        x, y (numpy array): Coordenadas de la cuadrícula.
        mux, muy (float): Medias en x e y.
        sigma_x, sigma_y (float): Desviaciones estándar en x e y.
        weight (float): Peso de la gaussiana.

    Retorna:
        (numpy array): Imagen generada de la gaussiana.
    """
    return weight * np.exp(-((x - mux) ** 2 / (2 * sigma_x ** 2) + (y - muy) ** 2 / (2 * sigma_y ** 2)))


def reconstruct_image_from_gmm(image_shape, means, covariances, weights):
    """
    Reconstruye la imagen a partir de los parámetros del GMM.
    
    Parámetros:
        image_shape (tuple): La forma de la imagen (ancho, alto).
        means (numpy array): Medias de las gaussianas.
        covariances (numpy array): Covarianzas de las gaussianas.
        weights (numpy array): Pesos de mezcla de las gaussianas.
    
    Retorna:
        reconstructed_image (numpy array): La imagen reconstruida.
    """
    # Crear una cuadrícula de coordenadas
    x_grid, y_grid = np.meshgrid(np.arange(image_shape[0]), np.arange(image_shape[1]), indexing='ij')

    # Inicializar la imagen reconstruida
    reconstructed_image = np.zeros(image_shape)

    # Iterar sobre las gaussianas para reconstruir la imagen
    for i in range(len(means)):
        mux, muy = means[i]  # Las coordenadas de la media (x, y)
        sigma_x, sigma_y = np.sqrt(covariances[i][0, 0]), np.sqrt(covariances[i][1, 1])  # Desviaciones estándar
        weight = weights[i]  # Peso de la gaussiana

        # Generar la gaussiana y añadirla a la imagen reconstruida
        gaussian = gaussian_2d(x_grid, y_grid, mux, muy, sigma_x, sigma_y, weight)
        reconstructed_image += gaussian

    return reconstructed_image

def plot_comparison(original_image, reconstructed_image):
    """
    Grafica la imagen original y la imagen reconstruida una al lado de la otra.

    Parámetros:
        original_image (numpy array): Imagen original.
        reconstructed_image (numpy array): Imagen reconstruida.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Imagen original
    axes[0].imshow(original_image, cmap='viridis')
    axes[0].set_title("Imagen Original")
    axes[0].axis('off')
    
    # Imagen reconstruida
    axes[1].imshow(reconstructed_image, cmap='viridis')
    axes[1].set_title("Imagen Reconstruida")
    axes[1].axis('off')
    
    plt.show()

# -------- GMM ------------
def gmm_vectors(img, mean, std, threshold=3, n_gaussians_positive=20, n_gaussians_negative=10):
    """
    Aplica modelos de mezclas gaussianas (GMM) a una imagen procesada por umbralización, generando vectores de medias, covarianzas y pesos.

    Parámetros:
        img (numpy array): Imagen a procesar (2D).
        mean (float): Media utilizada para la normalización de la imagen.
        std (float): Desviación estándar utilizada para la normalización de la imagen.
        threshold (float, opcional): Umbral para separar las frecuencias (por defecto 3).
        n_gaussians_positive (int, opcional): Número de componentes gaussianas positivas (por defecto 20).
        n_gaussians_negative (int, opcional): Número de componentes gaussianas negativas (por defecto 10).
    
    Retorna:
        combined_means (numpy array): Vectores de medias combinadas de los GMM positivo y negativo.
        combined_covariances (numpy array): Matriz de covarianzas combinadas de los GMM positivo y negativo.
        combined_weights (numpy array): Pesos combinados de los GMM positivo y negativo.
    """

    img_clear_positive, img_clear_negative = apply_threshold(img, mean, std, threshold)

    points = generate_points_from_image(img_clear_positive)
    gmm = GaussianMixture(n_components=n_gaussians_positive, covariance_type='full').fit(points)
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_

    negative_points = generate_points_from_image(-img_clear_negative)
    gmm_negative = GaussianMixture(n_components=n_gaussians_negative, covariance_type='full').fit(negative_points)
    means_negative = gmm_negative.means_
    covariances_negative = gmm_negative.covariances_
    weights_negative = -gmm_negative.weights_

    # ---- concatenar ambos resultados

    # Concatenar medios, covarianzas y pesos
    combined_means = np.vstack([means, means_negative])
    combined_covariances = np.vstack([covariances, covariances_negative])
    combined_weights = np.hstack([weights, weights_negative])

    return combined_means, combined_covariances, combined_weights

def gmm_batch_vectors(batch, n_gaussians_positive=30, n_gaussians_negative=10, threshold=2):
    """
    Aplica modelos de mezclas gaussianas (GMM) a un batch de imágenes, generando vectores de medias, desviaciones estándar y pesos para cada imagen.

    Parámetros:
        batch (numpy array): Batch de imágenes a procesar (batch_size, altura, anchura, 1).
        n_gaussians_positive (int, opcional): Número de componentes gaussianas positivas (por defecto 30).
        n_gaussians_negative (int, opcional): Número de componentes gaussianas negativas (por defecto 10).
        threshold (float, opcional): Umbral para separar las frecuencias (por defecto 2).
    
    Retorna:
        combined_batch (numpy array): Batch de vectores combinados de medias, desviaciones estándar y pesos de cada imagen,
                                      con shape (batch_size, n_gaussianas, 5). El vector tiene el formato 
                                      [mean_x, mean_y, std_x, std_y, weight] para cada gaussiana.
    """

    batch_size = batch.shape[0]
    combined_batch = []
    mean, std = calculate_image_stats(batch)  

    for i in range(batch_size):
        img = batch[i, :, :, 0]  # Tomamos cada imagen del batch
  
        # Aplicar umbrales y calcular GMMs
        img_clear_positive, img_clear_negative = apply_threshold(img, mean, std, threshold)

        # Verificar si hay puntos en la parte positiva
        points = generate_points_from_image(img_clear_positive)
        if len(points) == 0:
            means = np.zeros((n_gaussians_positive, 2))
            covariances = np.eye(2).reshape((1, 2, 2)).repeat(n_gaussians_positive, axis=0)
            weights = np.zeros(n_gaussians_positive)
        else:
            gmm = GaussianMixture(n_components=n_gaussians_positive, covariance_type='full').fit(points)
            means = gmm.means_
            covariances = gmm.covariances_
            weights = gmm.weights_

        # Verificar si hay puntos en la parte negativa
        negative_points = generate_points_from_image(-img_clear_negative)
        if len(negative_points) == 0:
            means_negative = np.zeros((n_gaussians_negative, 2))
            covariances_negative = np.eye(2).reshape((1, 2, 2)).repeat(n_gaussians_negative, axis=0)
            weights_negative = np.zeros(n_gaussians_negative)
        else:
            gmm_negative = GaussianMixture(n_components=n_gaussians_negative, covariance_type='full').fit(negative_points)
            means_negative = gmm_negative.means_
            covariances_negative = gmm_negative.covariances_
            weights_negative = -gmm_negative.weights_

        # Concatenar ambos resultados
        combined_means = np.vstack([means, means_negative])
        combined_covariances = np.vstack([covariances, covariances_negative])
        combined_weights = np.hstack([weights, weights_negative])

        # Extraer las desviaciones estándar (componentes diagonales de las covariancias)
        std_x = np.sqrt(np.maximum(combined_covariances[:, 0, 0], 1e-6))  # Desviación estándar en el eje x
        std_y = np.sqrt(np.maximum(combined_covariances[:, 1, 1], 1e-6))  # Desviación estándar en el eje y

        # Concatenar todos los valores relevantes en un vector (mean_x, mean_y, std_x, std_y, weight)
        combined_vectors = np.column_stack([combined_means[:, 0],  # mean_x
                                            combined_means[:, 1],  # mean_y
                                            std_x,                 # std_x
                                            std_y,                 # std_y
                                            combined_weights])     # weight

        combined_batch.append(combined_vectors)

    # Convertir la lista en un numpy array con shape (batch_size, n_gaussianas, 5)
    combined_batch = np.array(combined_batch)
    
    return combined_batch
