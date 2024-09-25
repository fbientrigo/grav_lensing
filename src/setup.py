import os
from setuptools import setup, find_packages

# Obtener la ruta absoluta de requirements.txt
current_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_directory, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

setup(
    name='grav_lens',
    version='0.3',
    packages=find_packages(),  # Encuentra automáticamente todos los paquetes en el directorio actual
    include_package_data=True,  # Incluye otros archivos como datos estáticos
    package_data={
        'grav_lens.models': ['*.pkl'],  # Incluye archivos .pkl en grav_lens/models
    },
    install_requires=requirements,  # Instala los paquetes listados en requirements.txt
)
