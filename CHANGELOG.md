# Changelog

Todos los cambios notables de este proyecto se documentarán en este archivo.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es/1.1.0/), y el proyecto se adhiere a [Versionado Semántico](https://semver.org/lang/es/).

## [Unreleased]

### Añadido
- Archivo `AGENTS.md` para agentes de IA con contexto del proyecto.
- Directorio `planning/` con documentación interna de desarrollo.
- Tres pruebas de regresión para los criterios de SQRT-ETmax con
  muestras positivas y mixtas y la conservación del AIC de Normal.
- Dependencias de pruebas reproducibles con un commit fijo de
  sqrt_etmax, utilizado también en CI.
- Campo `fit_method` en resultados personalizados y diez pruebas para
  selección del estimador, alias, valor predeterminado y entradas inválidas.
- Registro `fit_errors` por candidato, con tipo y mensaje del error.
- Nueve pruebas de regresión para fallos, reintentos, actualización de
  diferencias de AICc, ausencia de candidatos y ajustes personalizados.

### Corregido
- La ruta personalizada de SQRT-ETmax utiliza la log-verosimilitud
  mixta compartida para calcular AIC, AICc y BIC, evitando valores
  infinitos causados por evaluar logpdf en los ceros (#9).
- `"mle"` ejecuta máxima verosimilitud y `"lmoments"` se selecciona
  explícitamente. Los métodos inválidos producen ValueError antes del
  ajuste, sin quedar absorbidos por el manejador general de errores (#11).
- Los ajustes fallidos eliminan resultados anteriores del candidato y
  actualizan las diferencias de AICc de los restantes (#13).
- Los reintentos exitosos eliminan el error registrado anteriormente.
- El ranking admite un estado vacío y la selección sin ajustes disponibles
  produce un RuntimeError explícito.

### Cambiado
- Renombrado `AGENT.md` a `AGENTS.md` (consistencia con otros repositorios).
- Corregido copyright en `LICENSE`.
- `custom_type` utiliza `"mle"` por defecto; `"mel"` conserva su compatibilidad.

## [1.2.0] - 2026-08-06

### Añadido
- Función de utilidad `pbias_desglosado` para calcular el sesgo porcentual (PBIAS) total, de omisión y de comisión.
- Pruebas unitarias para `pbias_desglosado`.
- Workflow de publicación automática en PyPI (`.github/workflows/publish.yml`).
- Autor y licencia en `pyproject.toml`.

### Cambiado
- `pbias_desglosado` definido como método estático (`@staticmethod`).
- Eliminado print de depuración en `distCompare`.

## [1.1.0] - 2026-04-20

### Corregido
- Corrección de longitud de parámetros en distribuciones Log-Normal y SQRT-ETmax.
- Compatibilidad con Python 3.8.

## [1.0.0] - 2026-04-15

### Cambiado
- Corrección de la lógica de `get_best_dist()` y actualización de documentación previa al merge.
- Añadida trazabilidad en selección de modelos (campo `transp`).

## [0.1.0] - 2026-03-24

### Añadido
- Primera versión oficial con ajuste de distribuciones, tests y documentación.
