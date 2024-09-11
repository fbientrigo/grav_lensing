# La idea principal es:
```
y_data (128x128x1) -> parametros (5xNg)
```

Donde `Ng` corresponderia al numero de gaussianas

## El approach parametrico consta de dos pasos importantes


1. Pasar los datos de y_data a una distribución, lo que implica pasar todo de 0 a 1 evitando valores negativos
2. Hacer limpieza de datos, ya sea tomando thresholds o combinación de FFT para pasar 
3. Discretizar como puntos proporcionalmente a la densidad encontrada
4. Fitear con una mezcla de gaussianas y asi generar los parametros

Una vez se tienen todos los parametros de una buena cantidad de datos, estos se guardan con el mismo nombre del y_data original pero esta vez el arreglo se compone de parametros para construir la Gaussiana.

___

Actuandoo ocmo un MLOps, vamos a crear una pipeline para trabajar los datos, inicialmente se compone de una imagen con una distribucion (-0.5, 1) aunque estos limites pueden ser ligeramente distintos para los ficheros, 
Por ello utilice un standard scaler para que tome los datos y se entrene con el traindataset, pasandolos a (0, 1)

```
from utils import CustomScaler

# --- Ejemplo de uso del CustomScaler ---
scaler = CustomScaler()

# Ajustar el scaler a los datos de entrenamiento
scaler.fit_scaler(train_dataset)

# Normalizar un batch de datos
for X_batch, y_batch in train_dataset.take(1):
    y_batch_scaled = scaler.transform(y_batch)
    print("Scaled batch:", y_batch_scaled.shape)

# Desescalar para volver a los valores originales
y_batch_original = scaler.inverse_transform(y_batch_scaled)
print("Original batch:", y_batch_original.shape)

# lo que quiero hacer es poder mapear todos los datos a este limite 0, 1
# pues train_dataset es un objeto PreFetch de tensorflow
train_dataset.map()
```

Luego ya que conozco la mean de los datos y la desviacion estandar, le hago un proceso de threshold para tomar los datos por encima de tel threshold y solo quedarme con los peaks mas importantes

Tu objetivo es crear una funcion que haga el scaling y que ademas haga este threshold con los valores de mean y sigma que tengo guardados.

De esta manera se los puedo pasar por batchs de manera eficiente. ya que una sola imagen de y_train tendria forma (batch, 128, 128, 1)

Despues para que sepas que viene mas adelante, es el proceso de discrretizar, ya que es una distribucion donde estara los peaks y lo demas sera 0,  alli se generaran puntos proporcional a la densidad, de manera que quede con un arreglo de puntos

anteriormente tenia esta funcion
```python
def generate_points_from_image(image, scale_factor=1000):
    """
    Genera puntos (x, y) a partir de una imagen, donde la cantidad de puntos es proporcional a la intensidad del pixel.
    
    Parámetros:
        image (numpy array): Imagen en formato numpy array (2D).
        scale_factor (int): Factor de escala para determinar cuántos puntos generar proporcionalmente al valor del pixel.
        
    Retorna:
        points (list): Lista de coordenadas (x, y) generadas a partir de los valores de los píxeles.
    """
    points = []
    # Recorrer la imagen y generar puntos proporcionales al valor del pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            value = image[i, j]
            if value > 0:  # Solo considerar los picos no nulos
                n_points = int(value * scale_factor)
                points.extend([(i, j)] * n_points)  # Agregar (i, j) proporcionalmente al valor del pixel
    
    return np.array(points)
```
Me gustaria que con tus conocimientos de MLOps pienses y consideres la escalabilidad del codigo, y ademas como se almacenara esta multitud de puntos

si se le pasa un batch al algoritmo
(32, 128, 128, 1)
debe de generarse una coleccion de puntos para cada imagen
(32, n_points)

De manera que esta coleccion de puntos alimenten un algoritmo de Gaussian Mixtures
```python
# --- Ajustar GMM a los puntos generados ---
from sklearn.mixture import GaussianMixture

n_gaussians = 10  # Número de gaussianas para ajustar
gmm = GaussianMixture(n_components=n_gaussians, covariance_type='full').fit(points)
```

