# Guía de Contribución

¡Gracias por tu interés en contribuir a `hidroModelSelect`! Todas las aportaciones son bienvenidas.

## ¿Cómo contribuir?

1. Haz un *fork* del repositorio.
2. Crea una rama para tu característica o corrección de error (`git checkout -b feature/nueva-caracteristica`).
3. Escribe tu código (asegúrate de seguir el estilo del proyecto y documentar tus cambios).
4. Escribe o actualiza las pruebas pertinentes en la carpeta `tests/`.
5. Asegúrate de que las pruebas pasan ejecutando `pytest`.
6. Haz *commit* de tus cambios (`git commit -m 'Añade nueva característica'`).
7. Sube los cambios a tu rama (`git push origin feature/nueva-caracteristica`).
8. Abre un *Pull Request* en GitHub.

## Pruebas Unitarias

Usamos `pytest` para garantizar el correcto funcionamiento del paquete. Antes de enviar tu código, instala las dependencias de desarrollo y corre las pruebas de esta manera:
```bash
pip install -e .[test]
pytest tests/
```

## Reporte de Errores

Si encuentras un error o un resultado matemático inesperado en el ajuste de una distribución estadística, por favor abre un *Issue* en GitHub incluyendo:

- Una descripción clara del problema.
- El comportamiento esperado según la literatura.
- Código o datos de ejemplo para reproducir el fallo.

¡Gracias por ayudar a mejorar las herramientas de hidrología!