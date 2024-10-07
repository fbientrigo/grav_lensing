Muchas de las funciones de perdida asumen una forma o interpretación de la data, por ello en el ambito de generación de imagenes a veces no funcionaran correctamente "out of the box"

### KL Divergence
$$
KL(P,Q)= \sum_x P(x)  \log(\frac{P(x)}{Q(x)})
$$

Interpreta $P$ como la distribucion verdadera, $Q$ como la predicción del modelo. Cada una de estas debe encontrarse entre 0 y 1 o será simplemente clipped.

En geometria de la información, esta implica una distancia estadistica;
En inferencia, representa la cantidad de información perdida utilizando $Q$ como un aproximado de $P$

En teoria de codigo representa una cantidad de bits extras, pero la definición es algo confusa y requiere más profundidad

* rangos:
	* un problema viene dado a que los datos tienen valores negativos, osea que a veces $P(x)<0$ de manera que resultará en fallas de la KLD como una funcion de perdida
### 