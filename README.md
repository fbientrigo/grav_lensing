# grav_lensing
A Python toy model for gravitational lensing with educative means

## Contribuir
Tip 1: apenas vayas a comenzar a trabajar, es recomendable hacer
```bash
git fetch
git pull
```
Para actualizar cambios a local, luego de hacer los cambios pertinentes no olvides
```bash
git add "ficheros"
git commit -m "mensaje"
git push
```

## Instalar
Se utiliza Python 3.9.5, luego crea un entorno virtual
```bash
python -m venv venv
```

El ultimo corresponde al nombre del entorno virtual, luego activalo en windows como

```
.\venv\Scripts\activate
```

Ahora a instalar lo necesario
```
pip install -r requirements
```

Si se desea utilizar las utilidades y trabajar con los scripts modularizados, ir a la carpeta `final`
```bash
cd final
pip install -e .
```
Se encuentra implementado scripts para cargar los datos en formato tensorflow y ademas utilizar prefetching para ahorrar memoria.

# Distribución de Ficheros
- `lib/`: El usuario debe ingresar aquí para utilizar el dataset y las funcionalidades de las librerías.
  - `lib/grav_lens/`: Contiene la librería principal para el manejo del modelo de lente gravitacional.
    - `configs/`: Contiene archivos de configuración como rutas de datos.
    - `metrics/`: Las distintas metricas de la competencia
    - `models/`: Contiene archivos para construir los modelos una vez ya ha pasado las fases de experimentación
    - `testing/`: Contiene funciones de medición de recursos
    - `utils/`: Contiene utilidades y funciones auxiliares, como la carga y visualización de datasets.
    - `preprocess/`: Incluye filtering y pre procesamiento para modelos como el de Gaussian Mixture
    
    - `__init__.py`: Archivo de inicialización para la librería `grav_lens`.
  - `scripts/`: Scripts que realizan operaciones específicas utilizando la librería `grav_lens`.
    - `visualize_dataset.py`: Script para visualizar el dataset de manera gráfica.
    - `setup.py`: Archivo de configuración para la instalación del proyecto.

- `01_pre_exp/`: Directorio de experimentación inicial.
- `02_convergenceshear/`: Directorio para estudios y pruebas relacionadas con convergencia y cizalladura.
- `03_training/`: Directorio para scripts y notebooks relacionados con el entrenamiento del modelo.
- `docs_lensing/`: Contiene documentación, libros, presentaciones, cálculos y pizarras.
- `papers_lectures/`: Contiene artículos y material académico para investigar el estado del arte.
- `venv/`: Entorno virtual para la ejecución del proyecto.
- `README.md`: Archivo de documentación principal del proyecto.
- `requirements.txt`: Archivo con las dependencias necesarias para ejecutar el proyecto.
- `starting_kit_dataton.ipynb`: Notebook inicial para la competencia de datatón.
- `.gitignore`: Archivo que especifica qué ficheros y directorios deben ser ignorados por Git.

# Scripts basicos
Para correr la mayoria de scripts el usuario debe entrar en la carpeta `final` que corresponde al software final entregable, esto con el entorno de ejecucion activado.

Obtener datos cargados en memoria
```
python .\grav_lens\utils\load_dataset.py
```
Es posible ajustar el script para incluir mas datos, ahora mismo `MAX_FILES` esta en 1000 datos, que toma unos 30 segundos en cargar a memoria, pero usar numeros mayores requerira de mas tiempo de carga. Es totalmente ajustable y puede ser cambiado para hacer training del modelo final.


# Objetivos
1. Generar un modelo en formatos `tf` o `onnx` para ser subidos a la pagina del concurso, considerar limitaciones tecnicas del servidor.
2. Proveer de metricas extras para el entrenamiento, como
    - metrica de distribucion de Kappa (compara los errores de distribucion generados)
3. Generar una documentacion saludable de las implementaciones





# Datos

Los datos se pueden descargar mediante un gestor (7GB cada uno app)
A continuación se dejan los links en una lista para que tengan fácil acceso a ellos:
- Entrenamiento:
    1. https://descargas.inf.santiago.usm.cl/train/1.tar.gz
    2. https://descargas.inf.santiago.usm.cl/train/2.tar.gz
    3. https://descargas.inf.santiago.usm.cl/train/3.tar.gz
    4. https://descargas.inf.santiago.usm.cl/train/4.tar.gz
    5. https://descargas.inf.santiago.usm.cl/train/5.tar.gz
    6. https://descargas.inf.santiago.usm.cl/train/6.tar.gz
    7. https://descargas.inf.santiago.usm.cl/train/7.tar.gz
    8. https://descargas.inf.santiago.usm.cl/train/8.tar.gz
- Validación:
    - https://descargas.inf.santiago.usm.cl/test_public.tar.gz


## Subir el modelo
Toda la info, ademas la plataforma para subir el modelo a prueba se enceuntra en
https://www.codabench.org/competitions/3583/
