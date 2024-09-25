
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture



# ----------- Funciones Basicas de las que dependen las otras ----------------
import cv2

def downscale_image(image, scale=0.5):
    """
    Escala la imagen a la mitad de su resolución.
    """
    height, width = image.shape[:2]
    new_dim = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)


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


def generate_points_from_image(image, n_samples=100, density_threshold=0.01, density_scaling=True):
    """
    Genera una cantidad fija de puntos (x, y) a partir de una imagen, donde la cantidad de puntos es proporcional a la
    intensidad del pixel y los puntos de mayor densidad tienen más probabilidades de ser seleccionados.

    Parámetros:
        image (numpy array): Imagen en formato numpy array (2D).
        n_samples (int): Número total de puntos a generar.
        density_threshold (float, opcional): Umbral para ignorar píxeles de baja densidad (por defecto 0.01).
        density_scaling (bool, opcional): Si True, se escalarán las intensidades de los píxeles para aumentar la probabilidad 
                                          de seleccionar puntos más densos (por defecto True).

    Retorna:
        points (numpy array): Array de coordenadas (x, y) generadas a partir de los valores de los píxeles.
    """
    # Verificar si la imagen está vacía (todos los valores son cero)
    if np.sum(image) == 0:
        # Si está vacía, devolver una lista vacía de puntos
        return np.array([])

    # Aplanar la imagen para simplificar el muestreo
    flattened_image = image.flatten()

    # Aplicar el umbral para ignorar píxeles de baja densidad
    flattened_image = np.where(flattened_image >= density_threshold, flattened_image, 0)

    # Verificar si la imagen sigue vacía después del umbral
    if np.sum(flattened_image) == 0:
        return np.array([])

    # Si se habilita el escalado por densidad, aumentar la importancia de los píxeles con más densidad
    if density_scaling:
        flattened_image = flattened_image ** 2  # Incrementar la importancia de los píxeles más densos

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

import cv2

# Nueva función para escalar las gaussianas
def scale_gaussian_parameters(means, covariances, scale):
    """
    Escala los parámetros de las gaussianas (medias y covarianzas) para ajustar la resolución original.
    
    Parámetros:
        means (numpy array): Medias (x, y) de las gaussianas.
        covariances (numpy array): Covarianzas (matriz 2x2) de las gaussianas.
        scale (float): Factor de escalado inverso para volver a la resolución original.
    
    Retorna:
        scaled_means (numpy array): Medias escaladas.
        scaled_covariances (numpy array): Covarianzas escaladas.
    """
    scaled_means = means / scale  # Escalar las medias
    scaled_covariances = covariances / (scale ** 2)  # Escalar las covarianzas (cuadrado del factor de escala)
    return scaled_means, scaled_covariances

# Función para aplicar el GMM en imágenes escaladas
def gmm_batch_vectors(batch, n_gaussians_positive=30, n_gaussians_negative=10, threshold=2, n_points=500, 
                      scale=0.5, density_threshold=0.05, density_scaling=True,
                      pos_reg_covar=1e-6, pos_tol=1e-3, neg_reg_covar=1e-6, neg_tol=1e-3):
    """
    Aplica modelos de mezclas gaussianas (GMM) a un batch de imágenes escaladas, generando vectores de medias, desviaciones estándar y pesos.
    Luego, reescala las gaussianas para que correspondan a la resolución original.

    Esta función primero reduce la resolución de las imágenes de entrada, aplica un modelo GMM para extraer las componentes gaussianas
    positivas y negativas, y finalmente reescala las gaussianas a la resolución original. Se usa para aproximar las distribuciones
    de alta y baja frecuencia en las imágenes.

    Parámetros:
        batch (numpy.ndarray): Batch de imágenes a procesar con shape (batch_size, altura, anchura, 1).
        n_gaussians_positive (int, opcional): Número de componentes gaussianas positivas (por defecto 30).
        n_gaussians_negative (int, opcional): Número de componentes gaussianas negativas (por defecto 10).
        threshold (float, opcional): Umbral para separar las frecuencias. Define qué partes de la imagen se consideran "positivas" o "negativas" (por defecto 2).
        n_points (int, opcional): Número de puntos a generar para el modelo GMM a partir de la imagen umbralizada (por defecto 500).
        scale (float, opcional): Factor de escalado de la imagen para reducir la resolución (por defecto 0.5).
        pos_reg_covar (float, opcional): Regularización aplicada a las covariancias de las gaussianas positivas (por defecto 1e-3).
        pos_tol (float, opcional): Tolerancia para la convergencia del GMM positivo (por defecto 1e-5).
        neg_reg_covar (float, opcional): Regularización aplicada a las covariancias de las gaussianas negativas (por defecto 1e-3).
        neg_tol (float, opcional): Tolerancia para la convergencia del GMM negativo (por defecto 1e-5).

    Retorna:
        numpy.ndarray: Batch de vectores combinados con shape (batch_size, n_gaussianas, 5). Cada vector tiene el formato 
                       [mean_x, mean_y, std_x, std_y, weight], donde mean_x y mean_y son las medias de la gaussiana,
                       std_x y std_y son las desviaciones estándar, y weight es el peso de la gaussiana.

    Notas:
        - Se aplica un escalado a las imágenes antes de calcular las gaussianas, lo que reduce la cantidad de detalles procesados.
        - Después de aplicar GMM, los parámetros gaussianos se reescalan a la resolución original.
        - Esta función es útil para capturar tanto las frecuencias altas como bajas de la imagen usando GMM en distintas resoluciones.

    Ejemplo:
        >>> batch = np.random.random((10, 128, 128, 1))
        >>> result = gmm_batch_vectors(batch, n_gaussians_positive=20, n_gaussians_negative=5, scale=0.5)
        >>> result.shape
        (10, 25, 5)
    """
    batch_size = batch.shape[0]
    combined_batch = []
    original_shape = batch.shape[1:3]  # Obtener la forma original (altura, anchura)
    mean, std = calculate_image_stats(batch)

    for i in range(batch_size):
        img = batch[i, :, :, 0]  # Tomamos cada imagen del batch

        # Escalar la imagen
        img_scaled = downscale_image(img, scale=scale)

        # Aplicar umbrales y calcular GMMs en la imagen escalada
        img_clear_positive, img_clear_negative = apply_threshold(img_scaled, mean, std, threshold)

        # Verificar si hay puntos en la parte positiva
        points = generate_points_from_image(img_clear_positive, n_samples=n_points,  density_threshold=density_threshold, density_scaling=density_scaling)
        if len(points) == 0:
            means = np.zeros((n_gaussians_positive, 2))
            covariances = np.eye(2).reshape((1, 2, 2)).repeat(n_gaussians_positive, axis=0)
            weights = np.zeros(n_gaussians_positive)
        else:
            gmm = GaussianMixture(n_components=n_gaussians_positive, covariance_type='full', 
                                  reg_covar=pos_reg_covar, tol=pos_tol).fit(points)
            means = gmm.means_
            covariances = gmm.covariances_
            weights = gmm.weights_

        combined_means = means
        combined_covariances = covariances
        combined_weights = weights

        # Solo calcular la parte negativa si n_gaussians_negative > 0
        if n_gaussians_negative > 0:
            negative_points = generate_points_from_image(-img_clear_negative)
            if len(negative_points) == 0:
                means_negative = np.zeros((n_gaussians_negative, 2))
                covariances_negative = np.eye(2).reshape((1, 2, 2)).repeat(n_gaussians_negative, axis=0)
                weights_negative = np.zeros(n_gaussians_negative)
            else:
                gmm_negative = GaussianMixture(n_components=n_gaussians_negative, covariance_type='full',
                                               reg_covar=neg_reg_covar, tol=neg_tol).fit(negative_points)
                means_negative = gmm_negative.means_
                covariances_negative = gmm_negative.covariances_
                weights_negative = -gmm_negative.weights_

            # Concatenar los resultados de las gaussianas negativas
            combined_means = np.vstack([combined_means, means_negative])
            combined_covariances = np.vstack([combined_covariances, covariances_negative])
            combined_weights = np.hstack([combined_weights, weights_negative])

        # Reescalar las gaussianas a la resolución original ---------------------
        combined_means, combined_covariances = scale_gaussian_parameters(combined_means, combined_covariances, scale)

        # Extraer las desviaciones estándar (componentes diagonales de las covariancias)
        std_x = np.maximum(np.sqrt(combined_covariances[:, 0, 0]), 1e-12)  # Desviación estándar en el eje x
        std_y = np.maximum(np.sqrt(combined_covariances[:, 1, 1]), 1e-12)  # Desviación estándar en el eje y

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

# -------------- reconstruction ------------------

# Función para reconstruir las imágenes de baja frecuencia a partir de los coeficientes PCA
def reconstruct_lowfreq_from_pca(principal_components, batch_size):
    """
    Reconstruye las imágenes de baja frecuencia a partir de los coeficientes PCA.

    Parámetros:
        principal_components (numpy array): Coeficientes de PCA con shape (batch, n_components).
        batch_size (int): El tamaño del batch.
    
    Retorna:
        lowfreq_images (numpy array): Imágenes de baja frecuencia reconstruidas con shape (batch, 128, 128, 1).
    """
    # Inverso del transformado PCA
    low_freq_vstack = ipca_low.inverse_transform(principal_components)
    
    # Convertir de vstack (batch * 128 * 128) a forma (batch, 128, 128, 1)
    lowfreq_images = low_freq_vstack.reshape(batch_size, 128, 128, 1)
    
    return lowfreq_images

# Función para reconstruir imágenes de alta frecuencia en batch a partir de las gaussianas
def reconstruct_highfreq_from_gmm(gaussians, batch_size, image_shape=(128, 128)):
    """
    Reconstruye las imágenes de alta frecuencia a partir de los parámetros del GMM para todo el batch.

    Parámetros:
        gaussians (numpy array): Parámetros del GMM de shape (batch, n_gaussians, 5).
        batch_size (int): Tamaño del batch.
        image_shape (tuple): La forma de la imagen de salida (ancho, alto), por defecto (128, 128).

    Retorna:
        highfreq_images (numpy array): Imágenes de alta frecuencia reconstruidas con shape (batch, 128, 128, 1).
    """
    highfreq_images = []
    
    for i in range(batch_size):
        # Obtener los parámetros de las gaussianas para esta imagen
        means = gaussians[i, :, :2]
        covariances = np.array([[[g[2]**2, 0], [0, g[3]**2]] for g in gaussians[i]])  # Construir la matriz de covarianzas a partir de std
        weights = gaussians[i, :, 4]

        # Reconstruir la imagen desde los parámetros del GMM
        reconstructed_image = reconstruct_image_from_gmm(image_shape, means, covariances, weights)
        
        # Añadir al batch
        highfreq_images.append(reconstructed_image.reshape(image_shape[0], image_shape[1], 1))  # Añadir canal de color
    
    return np.array(highfreq_images)

# Función para reconstruir el batch completo sumando ambas frecuencias
def reconstruct_batch_images(principal_components, gaussians, batch_size, image_shape=(128, 128)):
    """
    Reconstruye el batch completo de imágenes sumando las componentes de baja y alta frecuencia.

    Parámetros:
        principal_components (numpy array): Coeficientes de PCA de baja frecuencia con shape (batch, n_components).
        gaussians (numpy array): Parámetros del GMM de alta frecuencia de shape (batch, n_gaussians, 5).
        batch_size (int): Tamaño del batch.
        image_shape (tuple): La forma de la imagen de salida (ancho, alto), por defecto (128, 128).

    Retorna:
        batch_images (numpy array): Batch de imágenes reconstruidas con shape (batch, 128, 128, 1).
    """
    # Reconstruir la parte de baja frecuencia
    lowfreq_images = reconstruct_lowfreq_from_pca(principal_components, batch_size)
    
    # Reconstruir la parte de alta frecuencia
    highfreq_images = reconstruct_highfreq_from_gmm(gaussians, batch_size, image_shape)
    
    # Sumar las dos componentes para obtener las imágenes originales
    batch_images = lowfreq_images + highfreq_images
    
    return batch_images


