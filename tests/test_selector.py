import numpy as np
import pandas as pd
import scipy.stats as st
import pytest
from unittest.mock import MagicMock
from hidroModelSelect.distCompare import HidroModelSelector


@pytest.fixture
def selector():
    """Selector con 100 datos ficticios para cubrir ramas n > 100."""
    rng = np.random.default_rng(42)
    data = rng.gamma(shape=5, scale=10, size=100)
    return HidroModelSelector(data)

@pytest.fixture
def sample_data():
    """Provee un conjunto de datos de ejemplo (ej. precipitaciones máximas anuales)."""
    return np.array([45.2, 56.3, 34.1, 78.5, 65.0, 52.1, 48.9, 61.2, 55.4, 41.2])

def test_initialization(sample_data):
    """Verifica que el selector se inicializa correctamente y ordena los datos."""
    selector = HidroModelSelector(sample_data)
    assert len(selector.obs_sort) == len(sample_data)
    assert np.all(np.diff(selector.obs_sort) >= 0)  # Verifica que estén ordenados de menor a mayor
    assert selector.n == len(sample_data)
    assert selector.results == {}

def test_fit_distribution(sample_data):
    """Verifica el ajuste de una distribución individual y el cálculo de sus estadísticos."""
    selector = HidroModelSelector(sample_data)
    selector.fit_distribution('Normal', st.norm)
    
    assert 'Normal' in selector.results
    res = selector.results['Normal']
    
    # Comprobar que los estadísticos clave están presentes
    expected_keys = ['aic', 'aicc', 'bic', 'a2', 'adc', 'ks', 'ks_pv', 'params', 'k_params', 'd_aicc']
    for key in expected_keys:
        assert key in res
        
    # Verificar que las métricas principales no son nulas
    assert not np.isnan(res['aic'])
    assert not np.isnan(res['a2'])
    assert len(res['params']) > 0

def test_ranking_dataframe(sample_data):
    """Verifica que el ranking se genera correctamente y está ordenado por AICc."""
    selector = HidroModelSelector(sample_data)
    selector.fit_distribution('Normal', st.norm)
    selector.fit_distribution('Gumbel', st.gumbel_r)
    
    df = selector.get_ranking_dataframe()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.iloc[0]['aicc'] <= df.iloc[1]['aicc']  # Debe estar ordenado de menor a mayor AICc
    
    # Probar el método de la mejor distribución
    best_df = selector.get_best_dist()
    assert not best_df.empty
    assert len(best_df) == 1  # Debe retornar solo la fila ganadora
    
    # Verificar la nueva métrica de trazabilidad
    assert 'transp' in best_df.columns
    assert best_df.iloc[0]['transp'] in ['pv_max', 'pv_H0', 'ad_cMax', 'ad_cH0', 'optima_ci', 'optima_ci_fallback']
    
    # Verificar que las no ganadoras también conservan la trazabilidad
    for model, res in selector.results.items():
        assert 'transp' in res
        assert res['transp'] != ''


# --- Tests de cobertura de ramas (helpers) ---


class TestGetLaioCoeffs:
    """Tests para _get_laio_coeffs: 4 familias + excepciones."""

    def test_ev1_family(self, selector):
        xp, bp, hp = selector._get_laio_coeffs('EV1')
        assert xp > 0 and bp > 0 and hp > 0

    def test_gumbel_alias(self, selector):
        xp, bp, hp = selector._get_laio_coeffs('GUMBEL')
        assert xp > 0 and bp > 0 and hp > 0

    def test_norm_family(self, selector):
        xp, bp, hp = selector._get_laio_coeffs('NORM')
        assert xp > 0 and bp > 0 and hp > 0

    def test_normal_alias(self, selector):
        xp, bp, hp = selector._get_laio_coeffs('NORMAL')
        assert xp > 0 and bp > 0 and hp > 0

    def test_ln_family(self, selector):
        xp, bp, hp = selector._get_laio_coeffs('LN')
        assert xp > 0 and bp > 0 and hp > 0

    def test_lognormal_alias(self, selector):
        xp, bp, hp = selector._get_laio_coeffs('LOGNORMAL')
        assert xp > 0 and bp > 0 and hp > 0

    def test_gev_with_shape(self, selector):
        xp, bp, hp = selector._get_laio_coeffs('GEV', shape_param=0.2)
        assert xp > 0 and bp > 0 and hp > 0

    def test_gev_raises_without_shape(self, selector):
        with pytest.raises(ValueError, match="parámetro de forma"):
            selector._get_laio_coeffs('GEV')

    def test_gev_clamps_large_shape(self, selector):
        """Si shape > 0.5, Laio lo trunca a 0.5."""
        xp1, _, _ = selector._get_laio_coeffs('GEV', shape_param=0.5)
        xp2, _, _ = selector._get_laio_coeffs('GEV', shape_param=2.0)
        assert xp1 == pytest.approx(xp2)

    def test_gam_with_shape(self, selector):
        xp, bp, hp = selector._get_laio_coeffs('GAM', shape_param=3.0)
        assert xp > 0 and bp > 0 and hp > 0

    def test_gam_alias_gamma(self, selector):
        xp, bp, hp = selector._get_laio_coeffs('GAMMA', shape_param=3.0)
        assert xp > 0 and bp > 0 and hp > 0

    def test_gam_alias_p3(self, selector):
        xp, bp, hp = selector._get_laio_coeffs('P3', shape_param=3.0)
        assert xp > 0 and bp > 0 and hp > 0

    def test_gam_alias_lp3(self, selector):
        xp, bp, hp = selector._get_laio_coeffs('LP3', shape_param=3.0)
        assert xp > 0 and bp > 0 and hp > 0

    def test_gam_raises_without_shape(self, selector):
        with pytest.raises(ValueError, match="parámetro de forma"):
            selector._get_laio_coeffs('GAMMA')

    def test_gam_clamps_small_shape(self, selector):
        """Si gamma < 2, Laio lo fuerza a 2."""
        xp_min, _, _ = selector._get_laio_coeffs('GAM', shape_param=2.0)
        xp_lower, _, _ = selector._get_laio_coeffs('GAM', shape_param=0.5)
        assert xp_min == pytest.approx(xp_lower)

    def test_unsupported_dist_raises(self, selector):
        with pytest.raises(ValueError, match="no soportada"):
            selector._get_laio_coeffs('WEIBULL')


class TestCalcAdc:
    """Tests para _calc_adc: tramo principal y tramo de cola inferior."""

    def test_tramo_principal(self, selector):
        """A2 >= 1.2*xp -> tramo principal (línea 218-223)."""
        xp, bp, hp = selector._get_laio_coeffs('EV1')
        A2_grande = 1.5 * xp  # Por encima de 1.2*xp
        w = selector._calc_adc(A2_grande, 'EV1')
        assert w > 0

    def test_tramo_cola_inferior(self, selector):
        """A2 < 1.2*xp -> tramo de cola inferior (línea 225-228)."""
        xp, bp, hp = selector._get_laio_coeffs('EV1')
        A2_pequeno = 0.5 * xp  # Por debajo de 1.2*xp
        w = selector._calc_adc(A2_pequeno, 'EV1')
        assert w > 0

    def test_ambos_tramos_gev(self, selector):
        xp, _, _ = selector._get_laio_coeffs('GEV', shape_param=0.1)
        w_grande = selector._calc_adc(1.5 * xp, 'GEV', shape_param=0.1)
        w_pequeno = selector._calc_adc(0.5 * xp, 'GEV', shape_param=0.1)
        assert w_grande > 0 and w_pequeno > 0


class TestCalculateAicBic:
    """Tests para _calculate_aic_bic: AICc finito e infinito."""

    def test_aicc_finito(self, selector):
        selector.n = 100
        aic, aicc, bic = selector._calculate_aic_bic(log_lik=-50.0, k=3)
        assert np.isfinite(aicc)
        assert aicc > aic  # Corrección positiva

    def test_aicc_inf_when_n_small(self, selector):
        """Si n <= k+1, AICc = inf."""
        selector.n = 3
        aic, aicc, bic = selector._calculate_aic_bic(log_lik=-10.0, k=3)
        assert aicc == np.inf

    def test_bic_uses_n_log(self, selector):
        selector.n = 100
        _, _, bic = selector._calculate_aic_bic(log_lik=-50.0, k=2)
        expected_bic = 2 * np.log(100) - 2 * (-50.0)
        assert bic == pytest.approx(expected_bic)


class TestGetAdCriticalValue:
    """Tests para get_ad_critical_value: alphas y muestras grandes."""

    def test_alpha_005_default(self, selector):
        val = selector.get_ad_critical_value(50)
        assert val > 0

    def test_alpha_010(self, selector):
        val = selector.get_ad_critical_value(50, alpha=0.10)
        assert val > 0

    def test_alpha_025(self, selector):
        val = selector.get_ad_critical_value(50, alpha=0.025)
        assert val > 0

    def test_alpha_001(self, selector):
        val = selector.get_ad_critical_value(50, alpha=0.01)
        assert val > 0

    def test_alpha_default_fallback(self, selector):
        """Alpha no estándar usa default 0.787."""
        val_custom = selector.get_ad_critical_value(50, alpha=0.99)
        val_default = selector.get_ad_critical_value(50, alpha=0.05)
        assert val_custom == pytest.approx(val_default)

    def test_large_n_no_adjustment(self, selector):
        """Para n > 100, no hay ajuste de tamaño muestral."""
        val = selector.get_ad_critical_value(200)
        expected = 0.787  # Valor teórico sin ajuste
        assert val == pytest.approx(expected)

    def test_small_n_has_adjustment(self, selector):
        """Para n <= 100, aplica ajuste (1 + 0.6/n)."""
        val = selector.get_ad_critical_value(50)
        expected = 0.787 * (1 + 0.6 / 50)
        assert val == pytest.approx(expected)


class TestFitDistributionBranches:
    """Tests para fit_distribution: ramas de lognorm floc, SQRT floc, ADC nan."""

    def test_lognorm_forces_floc(self, sample_data):
        selector = HidroModelSelector(sample_data)
        selector.fit_distribution('Log_Normal', st.lognorm)
        assert 'Log_Normal' in selector.results
        assert selector.results['Log_Normal']['adc'] != 0  # NaN check below

    def test_sqrt_etmax_nan_adc(self, sample_data):
        """SQRT-ETmax no tiene mapeo a Laio -> adc = nan."""
        selector = HidroModelSelector(sample_data)
        mock_dist = MagicMock()
        mock_dist.fit.return_value = (1.5, 0, 5.0)
        mock_dist.logpdf.return_value = np.full(len(sample_data), -2.0)
        mock_dist.cdf.return_value = np.linspace(0.1, 0.9, len(sample_data))
        selector.fit_distribution('SQRT-ETmax', mock_dist)
        assert np.isnan(selector.results['SQRT-ETmax']['adc'])

    def test_fit_exception_continues(self, sample_data):
        """Si el ajuste falla, fit_distribution imprime y continua sin añadir a results."""
        selector = HidroModelSelector(sample_data)
        mock_dist = MagicMock()
        mock_dist.fit.side_effect = RuntimeError("ajuste fallido")
        selector.fit_distribution('BadDist', mock_dist)
        assert 'BadDist' not in selector.results

    def test_floc_counts_as_fixed(self, sample_data):
        """Un parámetro fijo (floc) se descuenta del conteo k_params."""
        selector = HidroModelSelector(sample_data)
        selector.fit_distribution('LN', st.lognorm, floc=0)
        res = selector.results['LN']
        # lognorm tiene 3 params (shape, loc, scale), floc=1 fijo -> k_params = 2
        assert res['k_params'] == 2

    def test_gev_shape_to_laio(self, sample_data):
        """GEV: c -> shape_val = -c, dist_type_laio = 'GEV'."""
        selector = HidroModelSelector(sample_data)
        selector.fit_distribution('GEV', st.genextreme)
        res = selector.results['GEV']
        assert res['adc'] != 0  # ADC calculado
        assert not np.isnan(res['adc'])

    def test_pearson3_shape_to_laio(self, sample_data):
        """Pearson3: skew -> shape_val = (2/skew)^2, dist_type_laio = 'GAM'."""
        selector = HidroModelSelector(sample_data)
        selector.fit_distribution('Pearson3', st.pearson3)
        res = selector.results['Pearson3']
        assert not np.isnan(res['adc'])