# hidroModelSelect

[![Tests](https://github.com/INUAMA/hydroModelSelect/actions/workflows/ci.yml/badge.svg)](https://github.com/INUAMA/hydroModelSelect/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/hidromodelselect)](https://pypi.org/project/hidromodelselect/)
[![Branch coverage](https://img.shields.io/badge/coverage-65%25-brightgreen)](docs/COVERAGE.md)

`hidroModelSelect` es una librería en Python diseñada para facilitar
la selección y comparación de modelos de distribución estadística
aplicados a la hidrología (ej. análisis de frecuencias
de precipitaciones extremas o caudales).

El constructor admite muestras unidimensionales, no vacías, de valores
reales convertibles a float y finitos. Las entradas inválidas producen
ValueError antes de intentar ajustes. Se rechazan números complejos y
observaciones enmascaradas, sin descartar datos silenciosamente.

Se admiten ceros y valores negativos finitos; cada distribución aplica
sus propias restricciones de soporte y ajuste.

El selector conserva una copia independiente de los datos: `obs`
mantiene el orden original y `obs_sort` contiene una copia ordenada.
Modificar posteriormente el array de entrada no altera estas copias.

## Características

- Ajuste de múltiples distribuciones estadísticas (Gumbel, GEV, Normal, Log-Normal, Pearson3, SQRT-ETmax).
- Cálculo riguroso de criterios de bondad de ajuste y selección de modelos:
  - Criterio de Información de Akaike (AIC y AICc)
  - Criterio de Información Bayesiano (BIC)
  - Estadístico de Anderson-Darling (A²) y su versión estandarizada (ADC) según Laio (2004).
  - Prueba de Kolmogorov-Smirnov.
- Herramientas de visualización integradas:
  - Curvas de Probabilidad Acumulada.
  - Gráficos Q-Q (Quantile-Quantile).
  - Gráficos de Niveles de Retorno (Return Levels).

## Instalación

```bash
pip install hidromodelselect
```

Para desarrollo y ejecución de pruebas:

```bash
git clone https://github.com/INUAMA/hydroModelSelect.git
cd hydroModelSelect
python -m pip install -r requirements-test.txt

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

### Método de ajuste genérico

En la ruta genérica (`is_custom=False`), `fit_distribution` utiliza
máxima verosimilitud por defecto. El argumento `method` admite
`"MLE"` y `"MM"`, sin distinguir mayúsculas y minúsculas.

El método se valida y se transmite explícitamente a `fit`, conservando
las demás opciones de ajuste. Los resultados y el ranking lo registran
en `fit_method` como `"mle"` o `"mm"`.

MM designa el método de momentos ordinarios, distinto de L-momentos.
Este registro identifica el método solicitado; no certifica la
convergencia ni la optimalidad global del ajuste, ni valida el uso
de criterios de información convencionales con estimaciones no MLE.

Un método inválido produce ValueError antes del ajuste y conserva
los resultados y errores registrados previamente.

### Parámetros libres y restricciones

`k_params` registra el número de parámetros estimados. Las opciones de
fijación cuyo valor es None no se descuentan del recuento; los valores
realmente fijados, incluido cero, sí se descuentan.

Por ejemplo:

- Normal sin restricciones o con `floc=None`: dos parámetros libres.
- Normal con `floc=0`: un parámetro libre, la escala.
- Lognormal con la forma fijada mediante `f0`, `fs` o `fix_s` y la
  localización fijada en cero por defecto: un parámetro libre, la escala.

El recuento incluye las restricciones predeterminadas del selector.
Estos cambios conservan el ajuste solicitado y corrigen el número de
parámetros utilizado en AIC, AICc y BIC.

AICc conserva la convención de valor no disponible (`+inf`) cuando
`n <= k_params + 1`. Pasar una opción de fijación con None no convierte
ese AICc en disponible.

### Evaluación de Anderson–Darling

El estadístico A² se calcula mediante los métodos `logcdf` y `logsf`
de la distribución, evaluados en las observaciones ordenadas.
Las distribuciones utilizadas deben proporcionar ambos métodos.

Esto permite conservar contribuciones de cola cuando la CDF redondea
a cero o uno, siempre que los métodos logarítmicos de la distribución
puedan representarlas. El selector no recorta las probabilidades.

Una probabilidad realmente nula produce una contribución infinita.
El control de estadísticas no finitas excluye ese ajuste y registra
el motivo en `fit_errors`.

La calibración del contraste, incluidos los casos con parámetros
estimados o distribuciones mixtas, requiere una validación separada.

## Gestión de fallos de ajuste

Los errores capturados durante un ajuste se registran en
`selector.fit_errors`, por nombre del candidato, con los campos
`error_type` y `message`.

Un intento fallido elimina cualquier resultado anterior de ese candidato
y recalcula las diferencias de AICc de los candidatos restantes.
Un reintento exitoso guarda el nuevo resultado y elimina su error anterior.
Los ajustes personalizados mediante L-momentos conservan el sufijo `_Lmom`
también en el registro de errores.

Si no hay ajustes disponibles, `get_ranking_dataframe()` devuelve un
DataFrame vacío y `get_best_dist()` lanza `RuntimeError`.

Los valores no admitidos de `custom_type` siguen produciendo `ValueError`
antes de iniciar el ajuste.

Antes de aceptar un ajuste se comprueban los parámetros, la escala,
la CDF, la log-verosimilitud y los criterios y estadísticos calculados.
Los resultados numéricamente inválidos activan el mismo tratamiento
que una excepción del ajuste: retirada del candidato, registro de la
causa en `fit_errors` y actualización de las diferencias de AICc.

Se distinguen los valores inválidos de las cantidades no disponibles:

- `aicc = +inf` representa AICc no definido cuando `n <= k_params + 1`.
- `d_aicc = NaN` indica que la diferencia de AICc no está disponible.
- `adc = NaN` se conserva cuando la corrección de Laio no está
  disponible para esa distribución.

Estas comprobaciones verifican validez numérica; no acreditan la
optimalidad del ajuste ni la calibración estadística del selector.

## Ejemplo completo: selección entre seis distribuciones

El siguiente ejemplo ajusta las seis distribuciones soportadas y selecciona la
mejor según el algoritmo jerárquico del paquete. Se generan datos sintéticos de
precipitación extrema (50 eventos) directamente en el script para que el
ejemplo sea autocontenido.

```python
import numpy as np
import scipy.stats as st
from hidroModelSelect import HidroModelSelector

# Datos sintéticos de precipitación extrema (50 eventos, mm)
rng = np.random.default_rng(seed=42)
data = rng.gamma(shape=5.0, scale=15.0, size=50)

# Inicializar el selector
selector = HidroModelSelector(data)

# --- 1. Distribuciones de scipy ------------------------------------------------
selector.fit_distribution('Gumbel', st.gumbel_r)
selector.fit_distribution('GEV', st.genextreme)
selector.fit_distribution('Normal', st.norm)
selector.fit_distribution('Log_Normal', st.lognorm, floc=0)
selector.fit_distribution('Pearson3', st.pearson3)

# --- 2. SQRT-ETmax (distribución externa opcional) ------------------------------
#     Requiere el commit de sqrt_etmax fijado en requirements-test.txt.
try:
    import sqrt_etmax
    selector.fit_distribution(
        'SQRT-ETmax',
        sqrt_etmax.sqrt_etmax,
        is_custom=True,
        custom_type='mle',
    )
except ImportError:
    print("sqrt_etmax no instalado; se omite esta distribución.")

# --- 3. Ver ranking completo ----------------------------------------------------
ranking = selector.get_ranking_dataframe()
print(ranking[['aicc', 'bic', 'ks_pv', 'ad_c', 'adc']])

# --- 4. Seleccionar la mejor distribución ----------------------------------------
mejor = selector.get_best_dist(criterion="aicc")
print("\nMejor distribución:")
print(mejor[["criterion", "metri", "ks_pv", "ad_c", "transp"]])

# --- 5. Inspeccionar la trazabilidad (transp) -----------------------
# Etiquetas de la última selección:
#   'Criterio no disponible' – Excluida por criterio de información no finito
#   'Falla KS'              – No cumple la condición KS
#   'Falla AD'              – Cumple KS, pero no la condición AD
#   'pv_max'                – Alternativa: ninguna cumple KS; mayor p-valor
#   'ad_cH0'                – Única candidata que cumple KS y AD
#   'ad_cMax'               – Alternativa: ninguna tras KS cumple AD;
#                             se elige la de menor ad_c entre ellas
#   'Falla Optimo'          – Cumple KS y AD, pero queda fuera de ΔCI <= 2
#   'Desempate AD'          – Está dentro de ΔCI <= 2, pero no gana el desempate
#   'optima_ci'             – Menor ad_c entre las candidatas con ΔCI <= 2
#   'optima_ci_fallback'    – Alternativa por error en la etapa de criterio:
#                             menor ad_c entre las que cumplen KS y AD
#
# Las etiquetas describen las reglas actuales. Ser seleccionada como
# alternativa no implica superar los controles ni acredita calibración.
for nombre, info in selector.results.items():
    print(f"  {nombre}: transp={info['transp']}")

# --- 6. PBIAS desglosado (utilidad adicional) -----------------------------------
sim_array = rng.gamma(shape=5.0, scale=14.5, size=50)
p_tot, p_omi, p_com = HidroModelSelector.pbias_desglosado(
    np.array(data), sim_array
)
print(f"\nPBIAS Total: {p_tot:.2f}%, Omisión: {p_omi:.2f}%, Comisión: {p_com:.2f}%")
```

La etiqueta `name` identifica al candidato en los resultados. En la
ruta genérica, las decisiones específicas de cada familia utilizan
`dist_obj.name`: cambiar la etiqueta conserva las condiciones de ajuste
y la corrección de Laio seleccionada.

Las familias `lognorm` y `sqrt_etmax` utilizan `floc=0` por defecto.
Un valor de `floc` proporcionado explícitamente tiene prioridad.

Las familias sin corrección de Laio implementada conservan `adc=NaN`,
aunque su etiqueta coincida con la de otra distribución.

La transformación ADC utiliza la agrupación de la ecuación 11 de
Laio (2004), manteniendo los coeficientes de su ecuación 13
(`h0 = 0.851`). En el tramo inferior, el factor final multiplica
toda la suma. Esta corrección afecta a `adc`; no modifica `ad_c`
ni las reglas de selección.

Para GEV, el parámetro `theta3` de Laio (2004, tabla 1) coincide
con `c` de `scipy.stats.genextreme`: se transmite sin invertir el
signo. Esta corrección afecta a `adc`, conservando los parámetros
ajustados, A², `ad_c` y las reglas de selección. No valida por sí
sola el dominio de las tablas ni sus condiciones de aplicación.

Referencia: https://doi.org/10.1029/2004WR003204


### Disponibilidad de ADC

El campo `adc_reason` explica por qué ADC no está disponible.
En esos casos, `adc` contiene NaN y se conservan el ajuste y sus
restantes estadísticas.

La implementación omite ADC para el método de momentos ordinarios
(MM), para formas GEV inferiores a -1 y, de forma conservadora,
para MLE genérico con forma GEV >= 0.5 o forma GAM <= 2.
Estos últimos casos requieren condiciones de estimación que esta
ruta genérica no garantiza; se incluyen los valores frontera en
la exclusión.

En GAM, el límite de forma 2 se aplica a los coeficientes
asintóticos. La corrección por tamaño muestral utiliza la forma
original.

Estas comprobaciones no constituyen una calibración del selector.
La selección actual utiliza `ad_c`, distinto del campo `adc`.

La disponibilidad de ADC también depende de qué parámetros se estiman:

- Normal, Gumbel, GEV y Pearson III: la corrección implementada
  requiere estimar todos los parámetros de la familia.
- Lognormal: requiere localización fijada y estimación de forma
  y escala. Por defecto se utiliza loc=0. Una localización fija
  distinta de cero representa un desplazamiento conocido;
  una localización estimada no cumple esta configuración.
- Las opciones de fijación con valor None representan parámetros
  libres y no activan la exclusión.

Cuando la configuración no está respaldada, se conserva el ajuste
con adc=NaN y una explicación en adc_reason. Esta comprobación
no acredita por sí sola las restantes condiciones de aplicación
ni la calibración del selector.

### Tamaño muestral para informar ADC

La librería adopta los siguientes mínimos conservadores:

- Normal, Gumbel y Lognormal compatible: 10 observaciones.
- GEV y Pearson III compatible: 20 observaciones.

Por debajo del mínimo se conserva el ajuste, con adc=NaN y una
explicación en adc_reason. Las restantes condiciones de
aplicabilidad siguen siendo necesarias.

Estos mínimos se apoyan en los tamaños estudiados por Laio (2004),
sección 4 y apéndice B; no son límites matemáticos de las fórmulas
ni garantizan por sí solos la calibración.

No se impone un máximo de 100 observaciones. Los tamaños superiores
exceden la malla de simulaciones examinada en esa referencia;
las expresiones conservan su límite asintótico.

Esta política afecta al campo adc. No modifica la selección basada
en ad_c.

## Cómo se selecciona el mejor modelo

`get_best_dist(criterion="aicc")` utiliza un único criterio de información
para todos los candidatos. Las opciones admitidas son `"aic"`, `"aicc"`
y `"bic"`. El valor predeterminado es `"aicc"`.

Antes de aplicar los filtros estadísticos, se excluyen de esa selección
los candidatos cuyo criterio no sea numérico y finito. Sus ajustes se
conservan en `results`, con `transp="Criterio no disponible"`.
Si no hay candidatos elegibles, se lanza `RuntimeError`.

Sobre los candidatos elegibles se aplica la selección jerárquica:

1. **KS**: se exige un p-valor ≥ 0.05. Si ninguno cumple, se selecciona
   el de mayor p-valor; si solo uno cumple, se selecciona ese candidato.
2. **AD corregido**: entre quienes superan KS, se aplica el valor crítico
   utilizado por el selector. Si ninguno cumple, se elige el de menor
   AD corregido entre quienes superaron KS; si solo uno cumple, se elige.
3. **Criterio de información**: entre quienes superan ambos filtros,
   se conservan los candidatos con ΔCI ≤ 2 respecto al menor valor
   del criterio solicitado. Se elige el de menor AD corregido.

Todos los candidatos que superan KS pasan por la comprobación AD,
aunque solo quede uno. Si ese candidato cumple AD, se registra
ad_cH0; si lo incumple, la política actual lo devuelve como
alternativa con ad_cMax.

Ser seleccionado como alternativa no significa superar los
controles. Esta corrección del flujo no calibra los contrastes
ni garantiza un nivel de significación global.

La fila devuelta incluye `criterion`, el valor utilizado en `metri`
y la trazabilidad en `transp`. Los campos `criterion` y `transp` de
`results` describen la última selección; los parámetros y estadísticas
de los ajustes se conservan.

`get_ranking_dataframe()` continúa ordenando por AICc, y `d_aicc`
continúa representando diferencias de AICc, independientemente del
criterio solicitado para seleccionar.

### Estado de selección y modo estricto

`get_best_dist(criterion="aicc", require_pass=False)` conserva
por defecto la política habitual, incluidas sus alternativas.

El candidato seleccionado registra `selection_status`:

- `passes_current_checks`: cumple las condiciones KS y AD actuales.
- `fallback`: alternativa porque ningún candidato cumple ambos controles.

Los demás candidatos reciben `None`. El campo `transp` explica
la ruta concreta. Un `optima_ci_fallback` cumple KS y AD, aunque
haya ocurrido un error posterior en la etapa del criterio.

Con `require_pass=True`, si ningún candidato cumple ambos controles,
se lanza `RuntimeError`. Los ajustes y sus estadísticas se conservan,
se actualiza la trazabilidad y no queda un ganador registrado.

`require_pass` admite únicamente los booleanos de Python True y False.
Una configuración inválida se rechaza antes de modificar el estado.

Estos estados describen las reglas implementadas; no acreditan
calibración estadística ni aceptación del modelo.

## Contribuciones

¡Todas las contribuciones son bienvenidas! Revisa nuestro archivo
`CONTRIBUTING.md` para conocer los pasos para contribuir al proyecto y cómo
reportar problemas matemáticos o bugs en el código.
