# hidroModelSelect

`hidroModelSelect` es una librería en Python diseñada para facilitar la selección y comparación de modelos de distribución estadística aplicados a la hidrología (ej. análisis de frecuencias de precipitaciones extremas o caudales).

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
pip install -e .[test]
```

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
#     Requiere: pip install sqrt_etmax
try:
    import sqrt_etmax
    selector.fit_distribution(
        'SQRT-ETmax',
        sqrt_etmax.sqrt_etmax,
        is_custom=True,
        custom_type='mel',
    )
except ImportError:
    print("sqrt_etmax no instalado; se omite esta distribución.")

# --- 3. Ver ranking completo ----------------------------------------------------
ranking = selector.get_ranking_dataframe()
print(ranking[['aicc', 'bic', 'ks_pv', 'ad_c', 'adc']])

# --- 4. Seleccionar la mejor distribución ----------------------------------------
mejor = selector.get_best_dist()
print("\nMejor distribución:")
print(mejor[['aicc', 'ks_pv', 'ad_c', 'transp']])

# --- 5. Inspeccionar la trazabilidad (transp) ------------------------------------
#     Cada distribución en el ranking recibe una etiqueta que indica en qué
#     etapa del filtro fue descartada o seleccionada:
#       'Falla KS'         – No superó el test de Kolmogorov-Smirnov (p < 0.05)
#       'Falla AD'         – Superó KS pero no Anderson-Darling corregido
#       'pv_max'           – Ninguna pasó KS; se eligió la de mayor p-valor
#       'pv_H0'            – Solo una pasó KS
#       'ad_cMax'          – Ninguna pasó AD*; se eligió la de menor ad_c
#       'ad_cH0'           – Solo una pasó AD*
#       'Desempate AD'     – Pasó AD* pero no ganó el criterio óptimo
#       'optima_ci'        – Ganadora final (menor AICc/BIC dentro de Δ ≤ 2)
#       'optima_ci_fallback' – Fallback por error en el cálculo de Δ
for nombre, info in selector.results.items():
    print(f"  {nombre}: transp={info['transp']}")

# --- 6. PBIAS desglosado (utilidad adicional) -----------------------------------
sim_array = rng.gamma(shape=5.0, scale=14.5, size=50)
p_tot, p_omi, p_com = HidroModelSelector.pbias_desglosado(
    np.array(data), sim_array
)
print(f"\nPBIAS Total: {p_tot:.2f}%, Omisión: {p_omi:.2f}%, Comisión: {p_com:.2f}%")
```

## Cómo se selecciona el mejor modelo

El algoritmo de selección (`get_best_dist()`) aplica tres filtros secuenciales:

1. **Kolmogorov-Smirnov (KS)**: se exige un p-valor ≥ 0.05. Si ninguna
   distribución cumple, se selecciona la de mayor p-valor. Si solo una cumple,
   se elige automáticamente.
2. **Anderson-Darling corregido (AD\*)**: de las que superan KS, el estadístico
   corregido debe ser menor o igual al valor crítico ajustado por tamaño muestral
   al 95% de confianza. Si ninguna cumple, se elige la de menor AD\* entre las
   que pasaron KS.
3. **Métrica óptima (AICc / BIC)**: de las que superan AD\*, se preservan
   aquellas cuya distancia Δ respecto a la mejor métrica es ≤ 2.0. Se usa AICc
   cuando n/k < 40 y BIC cuando n/k ≥ 40. El desempate se resuelve por menor
   AD corregido.

Cada distribución recibe una etiqueta de trazabilidad en el campo `transp` (ver
tabla en el ejemplo anterior).

## Contribuciones

¡Todas las contribuciones son bienvenidas! Revisa nuestro archivo
`CONTRIBUTING.md` para conocer los pasos para contribuir al proyecto y cómo
reportar problemas matemáticos o bugs en el código.
