import numpy as np
import pytest

from hidroModelSelect import HidroModelSelector
from sqrt_etmax import sqrt_etmax
from scipy.stats import norm


@pytest.mark.parametrize(
    "datos",
    [
        np.array([0., 0., .5, 1., 2., 4., 8., 12.]),
        np.array([.5, 1., 2., 4., 8., 12.]),
    ],
    ids=["mixta", "positiva"],
)
def test_criterios_sqrt_con_ceros_usan_verosimilitud_mixta(datos):
    selector = HidroModelSelector(datos)

    selector.fit_distribution(
        "SQRT-ETmax",
        sqrt_etmax,
        is_custom=True,
        custom_type="mel",
    )

    resultado = selector.results["SQRT-ETmax"]
    k, loc, scale = resultado["params"]

    assert loc == 0
    assert resultado["k_params"] == 2

    log_lik = sqrt_etmax.log_likelihood(
        datos, k=k, alpha=1.0 / scale,
    )
    assert np.isfinite(log_lik)

    n = datos.size
    p = 2
    aic = 2 * p - 2 * log_lik

    esperados = {
        "aic": aic,
        "aicc": aic + 2 * p * (p + 1) / (n - p - 1),
        "bic": p * np.log(n) - 2 * log_lik,
    }

    for criterio, esperado in esperados.items():
        assert resultado[criterio] == pytest.approx(esperado)

    assert resultado["d_aicc"] == pytest.approx(0.0)

def test_aic_normal_conserva_suma_de_logpdf():
    datos = np.array([.5, 1., 2., 4., 8., 12.])
    selector = HidroModelSelector(datos)
    selector.fit_distribution("Normal", norm)

    resultado = selector.results["Normal"]
    log_lik = np.sum(norm.logpdf(datos, *resultado["params"]))

    assert np.isfinite(log_lik)
    assert resultado["k_params"] == 2
    assert resultado["aic"] == pytest.approx(4 - 2 * log_lik)
