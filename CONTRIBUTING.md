# Guía de Contribución

¡Gracias por tu interés en contribuir a `hidroModelSelect`! Todas las aportaciones son bienvenidas.

## ¿Cómo contribuir?

1. Haz un *fork* del repositorio.
2. Crea una rama para tu característica o corrección de error (`git checkout -b feature/nueva-caracteristica`).
3. Escribe tu código (asegúrate de seguir el estilo del proyecto y documentar tus cambios).
4. Escribe o actualiza las pruebas pertinentes en la carpeta `tests/` y actualiza la documentación (`README.md`, `CHANGELOG.md`, etc.) si es aplicable.
5. Asegúrate de que las pruebas pasan ejecutando `pytest` localmente.
6. Haz *commit* de tus cambios usando mensajes descriptivos y claros (`git commit -m 'feat: Añade nueva característica'`).
7. Sube los cambios a tu rama en GitHub (`git push origin feature/nueva-caracteristica`).
8. Abre un *Pull Request* hacia la rama principal.

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

## Preparación de Nuevas Versiones (Release Checklist)

Antes de etiquetar y publicar una nueva versión del paquete, asegúrate de revisar y actualizar los siguientes puntos para evitar desincronizaciones:

- [ ] **`pyproject.toml`**: Actualizar la variable `version` (siguiendo el estándar Semantic Versioning).
- [ ] **`CHANGELOG.md`**: Añadir una nueva sección con la fecha y versión de la release, documentando qué se ha añadido (`Added`), cambiado (`Changed`), deprecado (`Deprecated`) o arreglado (`Fixed`).
- [ ] **`README.md`**: Asegurar que los ejemplos de "Uso Rápido" siguen funcionando e incluir ejemplos de la nueva funcionalidad si es relevante.
- [ ] **`AGENT.md`**: Actualizar los ejemplos de uso y descripciones técnicas para mantener el contexto de los Agentes/LLMs al día.
- [ ] **`tests/`**: Comprobar que todos los tests pasan con éxito localmente (`pytest tests/`) y que se han añadido pruebas para el nuevo código.
- [ ] **`src/hidroModelSelect/__init__.py`**: Si se ha añadido una nueva clase o función principal, comprobar que se haya expuesto correctamente en la lista `__all__`.

Una vez completada esta lista, puedes proceder a hacer el *commit* de la release, añadir la etiqueta (ej. `git tag -a v1.2.0 -m "Release v1.2.0"`) y hacer el *push* correspondiente.