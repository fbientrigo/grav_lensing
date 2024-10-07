# Análisis de Eficiencia de Métodos de Carga de Datos en TensorFlow

## Resumen

En este análisis, se evaluó la eficiencia de dos métodos diferentes para cargar grandes conjuntos de datos en TensorFlow. Los métodos comparados fueron:

1. **Generadores Clásicos de TensorFlow**: Este enfoque carga todos los archivos `.npy` en la memoria antes de generar el conjunto de datos.
2. **Generadores Directos con Arrays de NumPy**: Este método utiliza una función generadora para cargar archivos `.npy` sobre la marcha, sin cargarlos todos en la memoria al mismo tiempo.

Se midió el rendimiento de cada método en términos de tiempo de carga y uso de memoria al manejar conjuntos de datos de tamaño creciente.

## Resultados

### Generadores Clásicos de TensorFlow Después de Cargar Arrays de NumPy

**Observaciones**:
- A medida que aumenta el número de archivos, tanto el tiempo como el uso de memoria aumentan significativamente.
- El proceso se vuelve mucho más lento y consume más memoria al manejar conjuntos de datos más grandes.

**Informe de Eficiencia**:

| Número Máximo de Archivos | Tiempo (segundos) | Memoria (MB) |
|---------------------------|-------------------|--------------|
| 100                       | 0.16              | 526.74       |
| 500                       | 0.13              | 565.51       |
| 1000                      | 0.17              | 611.45       |
| 5000                      | 6.19              | 963.48       |
| 10000                     | 10.23             | 1404.42      |
| 50000                     | 90.05             | 4921.11      |
| 70000                     | 134.08            | 6675.45      |

### Generadores Directos con Arrays de NumPy

**Observaciones**:
- El tiempo para cargar el conjunto de datos se mantiene bajo y casi constante, independientemente del número de archivos, gracias a la carga sobre la marcha.
- El uso de memoria es significativamente menor al manejar grandes conjuntos de datos, ya que solo se mantiene en memoria un pequeño lote de datos a la vez.

**Informe de Eficiencia**:

| Número Máximo de Archivos | Tiempo (segundos) | Memoria (MB) |
|---------------------------|-------------------|--------------|
| 100                       | 0.19              | 6675.45      |
| 500                       | 0.09              | 6675.45      |
| 1000                      | 0.10              | 6675.45      |
| 5000                      | 0.10              | 6675.45      |
| 10000                     | 0.10              | 6675.45      |
| 50000                     | 0.39              | 554.99       |
| 70000                     | 0.22              | 571.36       |

## Conclusión

Los resultados muestran claramente que el uso de generadores directos con arrays de NumPy es más eficiente, especialmente para conjuntos de datos grandes. Este método reduce drásticamente tanto el tiempo de carga como el uso de memoria, permitiendo un procesamiento de datos más rápido y escalable.

Por el contrario, el enfoque clásico de cargar todos los arrays de NumPy en la memoria antes de crear el conjunto de datos de TensorFlow se vuelve prohibitivamente lento y consume mucha memoria a medida que el tamaño del conjunto de datos aumenta.

### Recomendaciones

- Para grandes conjuntos de datos, se recomienda utilizar generadores que carguen datos sobre la marcha para optimizar tanto el tiempo como el consumo de memoria.
- Reserve el método clásico para poder hacer comparasiones.
