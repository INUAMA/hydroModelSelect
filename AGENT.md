# Agent Documentation: hidroModelSelect

Este documento describe el funcionamiento general del paquete `hidroModelSelect`, diseñado para facilitar a los profesionales e investigadores en hidrología la selección del mejor modelo probabilístico para sus datos.

---

## Instalación

Actualmente el paquete se puede instalar desde el código fuente. Clona este repositorio y ejecuta:

```bash
pip install .
```

Si vas a modificar el código, se recomienda instalarlo en modo desarrollo:

```bash
pip install -e .
```

## Uso Básico

### 1. Importación y Ajuste de Modelos

```python
import numpy as np
import scipy.stats as st
from hidroModelSelect import HidroModelSelector

# 1. Datos de ejemplo (ej. precipitaciones máximas anuales)
datos = np.array([45.2, 56.3, 34.1, 78.5, 65.0, 52.1, 48.9, 61.2])

# 2. Inicializar el selector
selector = HidroModelSelector(datos)

# 3. Ajustar distribuciones
selector.fit_distribution('Gumbel', st.gumbel_r)
selector.fit_distribution('Normal', st.norm)

# 4. Obtener el ranking de los mejores modelos
ranking = selector.get_ranking_dataframe()
print(ranking)

```

## Ejecución de Pruebas (Tests)

El proyecto incluye una suite de pruebas unitarias basadas en `pytest` para garantizar el comportamiento correcto de .

Para correr las pruebas, primero instala las dependencias de desarrollo:

```bash
pip install -e .[test]
pytest tests/
```

## Contribuciones y Reporte de Errores

Si deseas contribuir al código o has encontrado algún comportamiento matemático anómalo, por favor revisa el archivo CONTRIBUTING.md para más detalles sobre cómo abrir un *Pull Request* o un *Issue*.
