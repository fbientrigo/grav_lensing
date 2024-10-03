Tras construir las funciones de utilidad en los pasos previos se tiene todo implementado en la libreria, lo suficiente para extraer la información parametrizada de las imagenes.

Los archivos principales son:
- 1002_colabFit_tpu.ipynb: contiene el ultimo mejor entrenamiento, el cual usa bloques convolucionales y residuales
- models/: contiene los distintos modelos, a mayor fecha el modelo es superior a los anteriores, solo se almacena el ultimo mejor modelo, estos son en objetos .keras para hacer el proceso más limpio y rapido de programar, pero es recomendable guardar la función para construir el modelo, puede estar completamente parametrizado y luego con las instrucciones de construcción solo necesitarias descargar los pesos o weights en .h5, ocupando espacio minimo
- D_predict.ipynb: toma un modelo pre entrenado y ocupa la funcionalidad de la libreria para generar las distintas predicciones y almancenarlas en data/RESULT

