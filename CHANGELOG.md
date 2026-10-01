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
- 39 pruebas de regresión para la validación numérica de ajustes y
  el tratamiento de AICc no definido.
- Veinte pruebas de regresión para identidad de distribuciones,
  restricciones de localización y selección de la corrección de Laio.
- 23 pruebas de regresión para validación, normalización e independencia
  de las observaciones del selector.
- Parámetro `criterion` en `get_best_dist`, con opciones `"aic"`,
  `"aicc"` y `"bic"`; `"aicc"` es el valor predeterminado.
- Registro del criterio utilizado y de los candidatos excluidos por
  no disponer de un valor finito.
- Catorce pruebas de regresión para la selección con un criterio común.
- Campo `fit_method` también en los ajustes genéricos y en el ranking,
  con valores normalizados `"mle"` o `"mm"`.
- Doce pruebas de regresión para el registro, envío y validación del
  estimador genérico y la conservación de ajustes previos.
- Once pruebas de regresión para el recuento de parámetros libres:
  restricciones con None, valores fijados, alias de forma de Lognormal
  y disponibilidad de AICc.
- Tres pruebas de regresión para el valor de referencia de ADC,
  la continuidad entre tramos y la conservación del tramo superior.
- Cuatro pruebas de regresión para la forma GEV positiva, negativa
  y nula, y una referencia numérica del ADC resultante.
- Campo `adc_reason` para explicar por qué no se calcula ADC,
  conservando el ajuste y sus restantes estadísticas.
- Catorce pruebas de regresión sobre restricciones de parámetros
  y disponibilidad de ADC.
- Mínimos conservadores para informar ADC: 10 observaciones para
  NORM/EV1 y 20 para GEV/GAM. Los ajustes por debajo del mínimo
  se conservan con adc=NaN y un motivo explícito (#37).
- Ocho pruebas de regresión para exclusiones por tamaño y
  disponibilidad en los mínimos inclusivos.

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
- Se rechazan parámetros no finitos, escalas no positivas, CDF no
  finitas o fuera de [0, 1], log-verosimilitudes no finitas y criterios
  o estadísticos numéricamente inválidos. Los fallos se registran en
  `fit_errors` y el candidato queda excluido de los resultados (#15).
- Las diferencias de AICc se calculan tomando como referencia el menor
  AICc finito. Los candidatos con AICc no definido reciben `d_aicc = NaN`,
  evitando restas entre infinitos.
- Las restricciones automáticas y la selección de la corrección de
  Laio dependen de la identidad de la distribución, no de su etiqueta.
  Se conserva la prioridad de las restricciones explícitas (#17).
- El constructor rechaza muestras vacías, escalares, multidimensionales,
  no finitas, complejas, no convertibles o con observaciones enmascaradas,
  mediante ValueError explícitos (#19).
- La selección utiliza el mismo criterio de información para todos los
  candidatos, eliminando la alternancia entre AICc y BIC según n/k (#21).
- Los candidatos sin un valor finito del criterio solicitado se excluyen
  de esa selección, conservando sus ajustes. Si ninguno resulta elegible,
  se produce un RuntimeError explícito.
- La ruta genérica valida `method` antes de iniciar el ajuste, utiliza
  MLE por defecto y transmite explícitamente el método normalizado.
  Los nombres y tipos inválidos producen ValueError sin modificar
  resultados previos ni el registro de errores (#23).
- Las opciones de fijación con valor None ya no descuentan parámetros
  libres en los ajustes genéricos. Se conserva el recuento de valores
  realmente fijados, incluido cero, corrigiendo las penalizaciones de
  AIC, AICc y BIC y la disponibilidad de AICc (#25).
- El estadístico Anderson–Darling utiliza directamente logcdf y logsf
  en las rutas genérica y personalizada, evitando el recorte artificial
  de las contribuciones de cola. Las probabilidades realmente nulas
  conservan su contribución infinita (#27).
- Corregida la agrupación del tramo inferior de la transformación ADC:
  el factor final multiplica toda la suma, conforme a la ecuación 11
  de Laio (2004). Se conservan los coeficientes y el tramo superior (#29).
- GEV transmite a ADC el parámetro de forma de SciPy sin invertir
  su signo, conforme a la parametrización de Laio (2004), tabla 1 (#31).
- La corrección muestral GAM utiliza el parámetro de forma original;
  el límite inferior de 2 se aplica a los coeficientes asintóticos.
- Se validan los parámetros de forma utilizados por las aproximaciones
  ADC y se rechazan valores GEV inferiores a -1.
- El ajuste omite ADC para MM y, de forma conservadora, para MLE
  genérico con forma GEV >= 0.5 o GAM <= 2 (#33).
- ADC no se calcula para Normal, Gumbel, GEV y Pearson III con
  parámetros fijados adicionales a la configuración de referencia.
- En Lognormal, ADC requiere localización fijada y forma y escala
  estimadas. Las opciones con valor None no cuentan como parámetros
  fijados. Los ajustes se conservan y adc_reason explica la
  indisponibilidad (#35).

### Cambiado
- Renombrado `AGENT.md` a `AGENTS.md` (consistencia con otros repositorios).
- Corregido copyright en `LICENSE`.
- `custom_type` utiliza `"mle"` por defecto; `"mel"` conserva su compatibilidad.
- Las observaciones se almacenan como una copia independiente de tipo
  flotante. `obs` conserva el orden original, `obs_sort` contiene otra
  copia ordenada y `n` procede de la muestra validada.

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
