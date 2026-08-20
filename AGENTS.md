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

1. **KS**: Filtrar distribuciones con p-valor del test KS ≥ 0.05
2. **AD***: De las que pasan KS, filtrar con Anderson-Darling corregido ≤ valor crítico al 95%
3. **Métrica óptima**: Si n/k < 40 usar AICc, si ≥ 40 usar BIC. Desempate por menor AD corregido. Umbral ΔCI ≤ 2.0

El campo `transp` en el resultado documenta la trazabilidad del proceso de decisión.

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

## Instalación y Pruebas

```bash
# Instalación en modo desarrollo
pip install -e .[test]

# Ejecutar pruebas
pytest tests/ -v

# Verificar construcción
python -m build
```

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
