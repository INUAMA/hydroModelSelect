# Informe de Cobertura de Ramas — `get_best_dist` y Helpers

**Fecha**: 25 agosto 2026
**Rama**: `test/branch-coverage`
**Versión del paquete**: `1.2.0` (sin cambios de versión en esta PR)

## Entorno

| Componente | Versión |
|------------|---------|
| Python | 3.12.3 |
| pytest | 9.1.1 |
| pytest-cov | 7.1.0 |

## Comando reproducible

```bash
pip install -e .[test]
pytest tests/ --cov=src/hidroModelSelect --cov-branch --cov-report=term-missing -v
```

## Resultados

### Conteo de tests

| Archivo | Tests |
|---------|-------|
| `tests/test_get_best_dist.py` | 10 |
| `tests/test_selector.py` | 38 |
| `tests/test_utils.py` | 3 |
| **Total** | **51** |

### Cobertura por archivo

| Archivo | Stmts | Miss | Branch | BrPart | Cobertura |
|---------|-------|------|--------|--------|-----------|
| `__init__.py` | 7 | 2 | 0 | 0 | 71% |
| `distCompare.py` | 297 | 107 | 80 | 2 | 65% |
| **Total** | **304** | **109** | **80** | **2** | **65%** |

### Ramas no cubiertas (distCompare.py)

| Líneas | Función | Razón |
|--------|---------|-------|
| `250-270` | `fit_distribution` (rama `else` L-momentos) | Requiere `sqrt_etmax` externo (dependencia opcional, no instala en CI) |
| `275->280` | `fit_distribution` (rama SQRT floc) | Inalcanzable: SQRT-ETmax entra por `is_custom=True`, nunca alcanza el bloque `else` |
| `470-504` | `plotCCAcum` | Función de plotting; excluida del alcance de cobertura de algoritmo |
| `512-556` | `plot_qq` | Función de plotting; excluida del alcance |
| `566-624` | `plot_return_levels` | Función de plotting; excluida del alcance |

### Exclusiones justificadas

- **Funciones de plotting** (~180 líneas): `plotCCAcum`, `plot_qq`, `plot_return_levels` dependen de `matplotlib` y son de naturaleza visual. Se excluyen del alcance de cobertura del algoritmo de selección.
- **Rama L-momentos** (20 líneas): Requiere `sqrt_etmax` como dependencia externa. La dependencia es opcional por diseño (no declarada en `pyproject.toml`), documentada en el README con `pip install sqrt_etmax`.
- **Rama SQRT floc** (rama inalcanzable): Cuando `name in ['SQRT-ETmax', 'SQRT_ETmax']` el código entra por `is_custom=True`, por lo que nunca alcanza el bloque `else` donde se fuerza `floc=0`. Esta rama es defensiva.

## Tests nuevos añadidos (37 tests)

### `test_get_best_dist.py` (+2)

| Test | Rama cubierta |
|------|---------------|
| `test_sin_columna_k_params` | Línea 400: fallback `len(params)` cuando no existe `k_params` |
| `test_bic_todas_n_mayor_40` | BIC como métrica cuando todas las dists tienen n/k ≥ 40 |

### `test_selector.py` (+35)

| Test | Rama cubierta |
|------|---------------|
| `test_ev1_family` | `_get_laio_coeffs` familia EV1 |
| `test_gumbel_alias` | Alias GUMBEL → EV1 |
| `test_norm_family` | Familia NORM |
| `test_normal_alias` | Alias NORMAL → NORM |
| `test_ln_family` | Familia LN |
| `test_lognormal_alias` | Alias LOGNORMAL → LN |
| `test_gev_with_shape` | GEV con shape_param |
| `test_gev_raises_without_shape` | GEV sin shape → ValueError |
| `test_gev_clamps_large_shape` | GEV truncación shape > 0.5 |
| `test_gam_with_shape` | GAM con shape_param |
| `test_gam_alias_gamma` | Alias GAMMA |
| `test_gam_alias_p3` | Alias P3 |
| `test_gam_alias_lp3` | Alias LP3 |
| `test_gam_raises_without_shape` | GAM sin shape → ValueError |
| `test_gam_clamps_small_shape` | GAM truncación gamma < 2 |
| `test_unsupported_dist_raises` | Tipo no soportado → ValueError |
| `test_tramo_principal` | `_calc_adc` A2 ≥ 1.2·xp |
| `test_tramo_cola_inferior` | `_calc_adc` A2 < 1.2·xp |
| `test_ambos_tramos_gev` | `_calc_adc` ambos tramos con GEV |
| `test_aicc_finito` | `_calculate_aic_bic` AICc finito |
| `test_aicc_inf_when_n_small` | `_calculate_aic_bic` n ≤ k+1 → inf |
| `test_bic_uses_n_log` | `_calculate_aic_bic` BIC = k·ln(n) - 2·ln(L) |
| `test_alpha_005_default` | `get_ad_critical_value` α=0.05 |
| `test_alpha_010` | α=0.10 |
| `test_alpha_025` | α=0.025 |
| `test_alpha_001` | α=0.01 |
| `test_alpha_default_fallback` | α no estándar → default 0.787 |
| `test_large_n_no_adjustment` | n > 100 → sin ajuste |
| `test_small_n_has_adjustment` | n ≤ 100 → ajuste (1 + 0.6/n) |
| `test_lognorm_forces_floc` | Lognorm fuerza floc=0 |
| `test_sqrt_etmax_nan_adc` | SQRT-ETmax → adc = nan |
| `test_fit_exception_continues` | Excepción en ajuste → print + continúa |
| `test_floc_counts_as_fixed` | floc cuenta como parámetro fijo |
| `test_gev_shape_to_laio` | GEV: c → shape_val = -c |
| `test_pearson3_shape_to_laio` | Pearson3: skew → (2/skew)² |
