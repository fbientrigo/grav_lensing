Para crear la documentación se utiliza `sphinx` y `sphinx_rtd_theme`, medinate un workflow, lo que hace posible agregar la documentación de manera facil

```
sphinx-apidoc -o doc .\src\
.\doc\make.bat html
```

Luego se realiza un push

