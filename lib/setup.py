from setuptools import setup, find_packages

# Leer el archivo requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='grav_lens',
    version='0.2',
    packages=find_packages(),  # Encuentra automáticamente todos los paquetes dentro de grav_lens
    include_package_data=True,  # Incluye otros archivos como datos estáticos si los tienes
    install_requires=required,  # Instala los paquetes listados en requirements.txt
)
