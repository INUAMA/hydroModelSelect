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

Puedes instalar el paquete directamente desde el código fuente:

```bash
pip install .
```

Para desarrollo y ejecución de pruebas:

```bash
pip install -e .[test]
```

## Uso Rápido

```python
import scipy.stats as st
from hidroModelSelect import HidroModelSelector

# Datos de ejemplo
data = [45.2, 56.3, 34.1, 78.5, 65.0, 52.1, 48.9, 61.2]

# Inicializar y ajustar
selector = HidroModelSelector(data)
selector.fit_distribution('Gumbel', st.gumbel_r)
selector.fit_distribution('GEV', st.genextreme)

# Ver resultados
print(selector.get_ranking_dataframe())

# Obtener la mejor distribución automáticamente según los criterios de selección
mejor_modelo = selector.get_best_dist()
print(mejor_modelo)
```

## Contribuciones

¡Todas las contribuciones son bienvenidas! Revisa nuestro archivo `CONTRIBUTING.md` para conocer los pasos para contribuir al proyecto y cómo reportar problemas matemáticos o bugs en el código.