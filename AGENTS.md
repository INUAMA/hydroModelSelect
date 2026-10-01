# AGENTS.md - hidroModelSelect

## Contexto del Proyecto

- **Nombre**: `hidroModelSelect`
- **Propósito**: Selección y comparación de modelos de distribución estadística para hidrología.
- **Framework**: Python, estructurado como un paquete con layout `src/`.
- **Licencia**: MIT
- **PyPI**: `pip install hidromodelselect`
- **Repositorio**: https://github.com/INUAMA/hydroModelSelect

## Dependencias

- `scipy>=1.7.0`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `matplotlib>=3.4.0`

## Estructura del Repositorio

```
hydroModelSelect/
├── src/hidroModelSelect/
│   ├── __init__.py          # Exporta HidroModelSelector + __version__
│   └── distCompare.py       # Clase HidroModelSelector (624 líneas)
├── tests/
│   ├── test_get_best_dist.py  # 8 escenarios del algoritmo de selección
│   ├── test_selector.py       # Tests de integración
│   └── test_utils.py          # Tests de pbias_desglosado
├── .github/workflows/
│   ├── ci.yml               # CI para Python 3.8-3.12
│   └── publish.yml          # Publicación a PyPI vía Trusted Publishing
├── pyproject.toml
├── README.md
├── LICENSE
├── CONTRIBUTING.md
└── CHANGELOG.md
```

## Distribuciones Soportadas

| Distribución | Tipo scipy | Notas |
|-------------|-----------|-------|
| Gumbel (EV1) | `scipy.stats.gumbel_r` | Extreme Value Type I |
| GEV | `scipy.stats.genextreme` | Generalized Extreme Value |
| Normal | `scipy.stats.norm` | — |
| Log-Normal | `scipy.stats.lognorm` | Se fuerza `floc=0` |
| Pearson Type III | `scipy.stats.pearson3` | — |
| SQRT-ETmax | `sqrt_etmax.sqrt_etmax` | Distribución custom (paquete externo) |

**Nota** Infmormación importante sobre el estado de sqrt_etmax (como proyecto paralelo): https://github.com/INUAMA/sqrt_etmax/tree/main ¡Ojo ya lo tenemos publicado en PyPI! lo podemos instalar como `pip install sqrt_etmax` Comprobar las versiones disponibles.

## Algoritmo de Selección Jerárquico (`get_best_dist`)

El algoritmo selecciona la mejor distribución en 3 etapas:

`get_best_dist(criterion="aicc")` utiliza un criterio común para todos
los candidatos: `"aic"`, `"aicc"` o `"bic"`.

Antes de aplicar los filtros, excluye de esa selección los candidatos
sin un valor finito del criterio solicitado. Sus ajustes se conservan.
Si no quedan candidatos elegibles, lanza `RuntimeError`.

La selección jerárquica aplica:

1. **KS**: Filtrar candidatos con p-valor ≥ 0.05.
2. **AD corregido**: Entre quienes superan KS, aplicar el valor crítico
   utilizado por el selector.
3. **Criterio común**: Entre quienes superan ambos filtros, conservar
   candidatos con ΔCI ≤ 2 y elegir el de menor AD corregido.

Se mantienen las reglas alternativas existentes cuando ningún candidato
supera un filtro y la selección directa cuando solo uno lo supera.

Los campos `criterion` y `transp` registran la última selección.
Los parámetros y estadísticas de los ajustes se conservan.

Los fallos capturados se registran en `fit_errors` y eliminan cualquier
resultado anterior del candidato. Tras retirar un resultado se recalculan
las diferencias de AICc. Un reintento exitoso limpia su error previo.
El ranking admite resultados vacíos; seleccionar sin ajustes disponibles
produce `RuntimeError`.

La validación numérica previa al almacenamiento también utiliza
`fit_errors`. Deben conservarse las convenciones de cantidades no
disponibles: AICc no definido como `+inf`, su diferencia como `NaN`
y ADC no disponible como `NaN`. Solo los AICc finitos intervienen
en el cálculo del mínimo de referencia.

## Métricas de Bondad de Ajuste

- **AIC / AICc**: Criterio de Información de Akaike (corregido para muestras pequeñas)
- **BIC**: Criterio de Información Bayesiano
- **A2**: Estadístico de Anderson-Darling
- **ADC**: Anderson-Darling corregido (omega, Laio 2004 Eq 11, Laio 2009 Eq 4)
- **KS**: Kolmogorov-Smirnov

## Reglas de Código

- Python 3.8+ (compatibilidad verificada en CI)
- Docstrings en español con formato Google
- Todo código nuevo debe incluir pruebas unitarias en `tests/`
- `pbias_desglosado` es un `@staticmethod` (no depende del estado de la instancia)
- En la ruta genérica, utilizar `dist_obj.name` para identificar la
  familia; reservar `name` para la etiqueta del candidato. Conservar
  las restricciones explícitas y comprobar que renombrar un candidato
  no altera su ajuste ni la corrección de Laio seleccionada.
- Validar las observaciones en el constructor antes de utilizarlas.
  Comprobar máscaras y complejos antes de la conversión a float.
  Conservar copias independientes para `obs` y `obs_sort`, derivar `n`
  de la muestra validada y mantener la admisión de negativos finitos.
- En la ruta genérica, validar `method` antes del manejador de errores
  de ajuste. Admitir MLE/MM sin distinguir mayúsculas y minúsculas,
  utilizar MLE por defecto y transmitir el método normalizado a `fit`.
- Registrar el método solicitado en `fit_method` para todos los ajustes
  exitosos, conservando el contrato de los estimadores personalizados.
- Una configuración inválida del método debe conservar los resultados
  y errores previamente registrados.
- Al contar parámetros libres, distinguir las opciones de fijación
  con valor None de las restricciones efectivas. Un valor cero sí
  puede fijar un parámetro; evitar comprobaciones basadas en su
  valor booleano. Conservar los alias de forma y las restricciones
  predeterminadas de cada familia.
- Calcular Anderson–Darling a partir de `logcdf` y `logsf`, sin recortar
  probabilidades. Ambos vectores corresponden a las observaciones
  ordenadas; invertir log-SF dentro de la suma.
- Conservar las contribuciones infinitas de probabilidades realmente
  nulas y su tratamiento mediante el control de estadísticas no finitas.
- Los objetos simulados de pruebas deben proporcionar respuestas
  coherentes para los métodos logarítmicos utilizados.
- Conservar los paréntesis del tramo inferior de ADC:
  `(x0 + b0 * term1) * term2`. Verificar referencia, continuidad
  y tramo superior al modificar esta transformación.
- Para GEV, transmitir `c` de SciPy a ADC como `theta3=c`.
  No confundir esta convención con otras que utilizan `xi=-c`.
  Conservar las regresiones de signo y referencia numérica.
- Separar la indisponibilidad prevista de ADC de los fallos de cálculo.
  Registrar `adc_reason` y conservar el ajuste cuando ADC no sea aplicable.
  Si se calcula ADC y resulta no finito, mantener el rechazo del ajuste.
- En GAM, limitar únicamente la forma usada en los coeficientes
  asintóticos; utilizar la forma original en la corrección muestral.
- Documentar las exclusiones conservadoras en las fronteras de forma.
  No presentar estas comprobaciones como calibración estadística.
- Comprobar la configuración de parámetros antes de calcular ADC.
  Normal, Gumbel, GEV y Pearson III requieren todos sus parámetros
  estimados; Lognormal requiere localización fijada y forma y
  escala libres.
- Distinguir None de un valor fijado, incluido cero. No inferir
  las restricciones únicamente a partir de los parámetros finales.
- Conservar el ajuste cuando ADC no sea aplicable y registrar
  el motivo en adc_reason.
- Aplicar los mínimos de ADC en la ruta de ajuste: NORM/EV1 requieren
  n>=10; GEV/GAM, n>=20. Conservar los ajustes excluidos y registrar
  el motivo en adc_reason.
- Presentar esos mínimos como política conservadora, no como
  garantía de calibración. No introducir un máximo de 100.
- Las pruebas destinadas a verificar el cálculo ADC deben cumplir
  sus condiciones de aplicabilidad, incluido el tamaño muestral.

## Instalación y Pruebas

```bash
# Instalación en modo desarrollo
python -m pip install -r requirements-test.txt

# Ejecutar pruebas
pytest tests/ -v

# Verificar construcción
python -m build
```
### Integración con SQRT-ETmax

La ruta personalizada (`is_custom=True`) requiere una versión de
`sqrt_etmax` que exponga `log_likelihood(data, k, alpha)`. El commit
probado está fijado en `requirements-test.txt`.

Esta ruta utiliza la log-verosimilitud mixta para calcular AIC, AICc
y BIC: los ceros aportan su probabilidad puntual y los valores
positivos su densidad. La ruta genérica conserva la suma de logpdf.

La corrección de estos cálculos no acredita por sí sola la calibración
de KS/AD ni la comparabilidad entre familias con distintos modelos de
observación para los ceros.

El estimador personalizado predeterminado es `custom_type="mle"`.
Se conserva `"mel"` como alias compatible; `"lmoments"` selecciona
explícitamente L-momentos. Los tipos o nombres no admitidos producen
`ValueError` antes del ajuste. Cada resultado personalizado registra
el estimador canónico en `fit_method`: `"mle"` o `"lmoments"`.
Este registro no valida el uso de criterios de información convencionales
con estimaciones obtenidas mediante L-momentos.


## Git Workflow

- Antes de empezar trabajo nuevo, crear (o localizar) un issue en GitHub que
  describa la tarea; issues y descripciones de PR se escriben en inglés, con
  el label apropiado (`enhancement`, `documentation`, `bug`, ...).
- Toda PR debe vincularse a su issue con `Closes #N` en la primera línea de
  su descripción.
- Crear una rama de trabajo antes de editar: `git switch -c <tipo>/<descripción>`
- No commitear directamente a `main`
- Hacer push de la rama de trabajo: `git push -u origin <branch>`
- Fusionar mediante Pull Request en GitHub
