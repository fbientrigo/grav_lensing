import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label, center_of_mass
from scipy import stats

def get_stats(data):
    """
    Obtiene la media y la desviación estándar de un conjunto de datos.

    Parameters:
        data (numpy.ndarray): Conjunto de datos para el cual calcular las estadísticas.

    Returns:
        tuple: Una tupla que contiene la media y la desviación estándar.

    Examples:
        >>> datos = np.array([1, 2, 3, 4, 5])
        >>> media, desviacion = get_stats(datos)
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    return mean, std_dev

def get_peaks(data, mean, std_dev, sigmas = 5):
    """
    Identifica los picos en un conjunto de datos que superan un umbral dado por la media más un múltiplo de la desviación estándar.

    Parameters:
        data (numpy.ndarray): Conjunto de datos en el cual buscar los picos.
        mean (float): Media del conjunto de datos.
        std_dev (float): Desviación estándar del conjunto de datos.
        sigmas (int, optional): Número de desviaciones estándar por encima de la media para considerar un pico. Por defecto es 5.

    Returns:
        list of tuples: Una lista de tuplas con las coordenadas de los picos identificados.

    Examples:
        >>> datos = np.random.random((10, 10))
        >>> media, desviacion = get_stats(datos)
        >>> picos = get_peaks(datos, media, desviacion, sigmas=3)
    """
    threshold = mean + sigmas * std_dev  # 5 sigma threshold
    peaks = data > threshold
    labeled, num_features = label(peaks)
    peak_indices = center_of_mass(data, labeled, range(1, num_features + 1))
    return peak_indices

def plot_peaks(data, sigmas=5, ax1=None, ax2=None):
    """
    Grafica la distribución de datos y destaca los picos que superan un umbral especificado.

    Parameters:
        data (numpy.ndarray): Conjunto de datos a graficar.
        sigmas (int, optional): Número de desviaciones estándar por encima de la media para destacar los picos. Por defecto es 5.
        ax1 (matplotlib.axes.Axes, optional): Eje de la primera gráfica (distribución con picos). Si no se proporciona, se crea uno nuevo.
        ax2 (matplotlib.axes.Axes, optional): Eje de la segunda gráfica (histograma de valores). Si no se proporciona, se crea uno nuevo.

    Returns:
        None

    Examples:
        >>> datos = np.random.random((10, 10))
        >>> plot_peaks(datos, sigmas=3)
    """
    if ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    mean, std_dev = get_stats(data)
    peak_idx = get_peaks(data, mean, std_dev, sigmas)

    # Plot the data with peaks highlighted on ax1
    im = ax1.imshow(data, origin='lower')
    ax1.set_title('Picos por encima de {} Sigmas'.format(sigmas))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Plot the peaks
    for peak in peak_idx:
        y, x = peak
        ax1.plot(x, y, 'rx', markersize=10, markeredgewidth=2)  # Cross marker

        # Draw confidence interval (e.g., 1 sigma circle)
        confidence_interval = plt.Circle((x, y), std_dev, color='r', fill=False, linestyle='--')
        ax1.add_patch(confidence_interval)

    # Añadir una barra de color
    plt.colorbar(im, ax=ax1, label='Densidad')

    # Graficar la distribución de valores en ax2
    ax2.hist(data.ravel(), bins=50, color='blue', alpha=0.7)
    ax2.axvline(mean, color='red', linestyle='dashed', linewidth=1, label='Media')
    ax2.axvline(mean + sigmas * std_dev, color='green', linestyle='dashed', linewidth=1, label=f'{sigmas} Sigma')
    ax2.set_title('Distribución de Valores')
    ax2.set_xlabel('Valor')
    ax2.set_ylabel('Frecuencia')
    ax2.legend()

    plt.tight_layout()
    plt.show()