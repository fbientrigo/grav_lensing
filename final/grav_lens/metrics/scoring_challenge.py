import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
import unittest


# ======= Metric used for scoring in the challenge ========

def WMAPE(T: tf.Tensor, P: tf.Tensor, delta=tf.constant(1e-5, dtype=tf.float32)):
    """
    Calcula el Error Medio Absoluto Porcentual Ponderado (WMAPE) entre dos tensores.

    El WMAPE es una métrica de error que tiene en cuenta la magnitud de las
    observaciones reales al ponderar las diferencias absolutas entre las predicciones
    y los valores reales. Esta métrica es útil en escenarios donde los valores reales
    tienen una variabilidad significativa, permitiendo un cálculo más equitativo del error.

    Parameters:
        T (tf.Tensor): Tensor que representa los valores reales u observados.
        P (tf.Tensor): Tensor que representa los valores predichos.
        delta (tf.Tensor, optional): Un pequeño valor constante para evitar divisiones por cero. 
                                     Por defecto es 1e-5.

    Returns:
        tf.Tensor: Un tensor escalar que representa el WMAPE calculado.

    Examples:
        >>> T = tf.constant([10.0, 0.0, 5.0])
        >>> P = tf.constant([9.0, 1.0, 5.0])
        >>> wmape = WMAPE(T, P)
        >>> print(wmape.numpy())
    """
    T = tf.where(T == 0.0, delta, T)
    P = tf.where(P == 0.0, delta, P)
    W_fun = lambda T, max_T: 1 + (T / max_T)
    max_T = tf.reduce_max(T)
    W = W_fun(T, max_T)
    Numerator = W * (tf.abs(P - T) / tf.abs(T))
    sum_W = tf.reduce_sum(W)
    sum_Numerator = tf.reduce_sum(Numerator)
    return sum_Numerator / sum_W

def DICE_binary_mask(tensor: tf.Tensor):
    """
    Genera una máscara binaria a partir de un tensor.

    Esta función crea una máscara binaria basada en la diferencia de cada elemento del tensor
    respecto a la media del tensor. Los valores mayores a la media se establecen en 1, mientras
    que los valores menores o iguales a la media se establecen en 0.

    Parameters:
        tensor (tf.Tensor): Un tensor de entrada sobre el cual se calculará la máscara binaria.

    Returns:
        tf.Tensor: Un tensor binario (del mismo tamaño que el tensor de entrada) con valores 1 para
                   los elementos que están por encima de la media y 0 para los que están por debajo
                   o son iguales a la media.

    Examples:
        >>> tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        >>> mask = DICE_binary_mask(tensor)
        >>> print(mask.numpy())
    """
    tensor_mask = tensor - tf.reduce_mean(tensor)
    tensor_mask = tf.where(tensor_mask > 0, tf.ones_like(tensor_mask), tf.zeros_like(tensor_mask))
    return tensor_mask

def DICEE(T: tf.Tensor, P: tf.Tensor, alpha: float = 0.5, beta: float = 0.5):
    """
    Calcula el coeficiente DICE ponderado entre dos tensores.

    El coeficiente DICE es una métrica de similitud utilizada principalmente en tareas de segmentación,
    donde se evalúa el grado de coincidencia entre dos conjuntos binarios. Esta función extiende el coeficiente
    DICE estándar al permitir ponderar falsos positivos y falsos negativos con los parámetros alpha y beta.

    Parameters:
        T (tf.Tensor): Tensor que representa las etiquetas reales o el ground truth.
        P (tf.Tensor): Tensor que representa las predicciones.
        alpha (float, optional): Factor de ponderación para los falsos negativos. Por defecto es 0.5.
        beta (float, optional): Factor de ponderación para los falsos positivos. Por defecto es 0.5.

    Returns:
        tf.Tensor or None: Un tensor escalar que representa el coeficiente DICE calculado.
                           Devuelve None si las formas de T y P no coinciden.

    Examples:
        >>> T = tf.constant([[1.0, 2.0], [0.0, 0.0]])
        >>> P = tf.constant([[1.0, 1.0], [0.0, 0.0]])
        >>> dice = DICEE(T, P)
        >>> print(dice.numpy())
    """
    if T.shape != P.shape:
        return None
    G = DICE_binary_mask(T)
    A = DICE_binary_mask(P)
    sum_G_A = tf.reduce_sum(G * A)
    sum_alpha = tf.reduce_sum(A * (1 - G))
    sum_beta = tf.reduce_sum(G * (1 - A))
    return 1 - (sum_G_A / (sum_G_A + alpha * sum_alpha + beta * sum_beta))

def find_top_k_peaks(im, sigma=3, N=3):
    smoothed = gaussian_filter(im, sigma=sigma)
    coordinates = peak_local_max(smoothed, threshold_abs=None, num_peaks=N)
    while len(coordinates) < 3:
        coordinates = np.vstack([coordinates, np.array([0, 0])])
    return coordinates

def DPEAKS(T: np.ndarray, P: np.ndarray, num_peaks=3):
    """
    Calcula la distancia absoluta sumada entre las cumbres principales de dos imágenes.

    Esta función compara las cumbres principales de dos imágenes encontradas mediante la función
    `find_top_k_peaks` y calcula la suma de las distancias absolutas entre las posiciones de cumbres
    correspondientes.

    Parameters:
        T (np.ndarray): Primera imagen (o tensor) para comparar.
        P (np.ndarray): Segunda imagen (o tensor) para comparar.
        num_peaks (int, optional): Número de cumbres principales a encontrar en cada imagen. Por defecto es 3.

    Returns:
        float: La suma de las distancias absolutas entre las cumbres principales de las dos imágenes.

    Examples:
        >>> T = np.random.rand(10, 10)
        >>> P = np.random.rand(10, 10)
        >>> dpeaks = DPEAKS(T, P, num_peaks=3)
        >>> print(dpeaks)
    """
    PEAKS_T = find_top_k_peaks(T, N=num_peaks)
    PEAKS_P = find_top_k_peaks(P, N=num_peaks)
    sum_DPEAKS = np.sum(np.abs(PEAKS_T - PEAKS_P))
    return sum_DPEAKS

# ======= Test the functions ========

class TestMetrics(unittest.TestCase):

    def setUp(self):
        # Example tensors for testing
        self.T = tf.constant([[2.0, 3.0], [5.0, 7.0]])
        self.P = tf.constant([[2.5, 3.5], [4.0, 6.0]])

    def test_WMAPE(self):
        result = WMAPE(self.T, self.P)
        expected = 0.1848148
        self.assertAlmostEqual(result.numpy(), expected, places=5)

    def test_DICE_binary_mask(self):
        tensor = tf.constant([[1.0, 2.0], [-1.0, -2.0]])
        result = DICE_binary_mask(tensor)
        expected = tf.constant([[1.0, 1.0], [0.0, 0.0]])
        self.assertTrue(np.array_equal(result.numpy(), expected.numpy()))

    def test_DICEE(self):
        T = tf.constant([[1.0, -1], [0.0, 0.0]])
        P = tf.constant([[1.0, 1.0], [0.0, 0.0]])
        result = DICEE(T, P)
        expected = 0.3333333
        self.assertAlmostEqual(result.numpy(), expected, places=5)

    def test_DPEAKS(self):
        center = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
            [1, 2, 3, 3, 3, 3, 3, 3, 2, 1],
            [1, 2, 3, 4, 4, 4, 4, 3, 2, 1],
            [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
            [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
            [1, 2, 3, 4, 4, 4, 4, 3, 2, 1],
            [1, 2, 3, 3, 3, 3, 3, 3, 2, 1],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])

        # Create a 10x10 tensor with peaks near the borders
        border = np.array([
            [0, 1,  1,  1,  1,  1,  1,  1,  8, 10],
            [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [10, 1,  1,  1,  1,  1,  1,  1,  1, 10]
        ])
        result = DPEAKS(center, border)
        expected = 15
        self.assertEqual(result, expected)

def main():
    print("Testing WMAPE")
    T = tf.constant([[10.0, 0.0], [0.0, 20.0]])
    P = tf.constant([[8.0, 1.0], [0.0, 25.0]])
    print("WMAPE:", WMAPE(T, P).numpy())

    print("Testing DICEE")
    T = tf.constant([[1.0, 2.0], [0.0, 0.0]])
    P = tf.constant([[1.0, 1.0], [0.0, 0.0]])
    print("DICEE:", DICEE(T, P).numpy())

    print("Testing DPEAKS")
    T = np.array([[1, 2], [3, 4]])
    P = np.array([[1, 2], [4, 3]])
    print("DPEAKS:", DPEAKS(T, P))

    # Run unit tests
    unittest.main(argv=[''], exit=False)

if __name__ == "__main__":
    main()
