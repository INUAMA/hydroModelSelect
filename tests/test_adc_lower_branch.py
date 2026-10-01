import numpy as np
import pytest

from hidroModelSelect import HidroModelSelector


def test_adc_tramo_inferior_coincide_con_referencia():
    selector = HidroModelSelector(np.arange(1.0, 51.0))

    # Laio (2004), ecuaciones 11 y 13.
    # Caso NORM, n=50, A²=xp.
    resultado = selector._calc_adc(0.168002, "NORM")

    assert resultado == pytest.approx(
        0.039083567841196376,
        rel=1e-12,
    )

def test_adc_es_continuo_en_union_de_tramos():
    selector = HidroModelSelector(np.arange(1.0, 51.0))
    xp, _, _ = selector._get_laio_coeffs("NORM")
    limite = 1.2 * xp

    izquierda = selector._calc_adc(
        np.nextafter(limite, -np.inf), "NORM"
    )
    en_limite = selector._calc_adc(limite, "NORM")
    derecha = selector._calc_adc(
        np.nextafter(limite, np.inf), "NORM"
    )

    assert izquierda == pytest.approx(en_limite, rel=1e-12)
    assert derecha == pytest.approx(en_limite, rel=1e-12)


def test_adc_conserva_tramo_superior():
    selector = HidroModelSelector(np.arange(1.0, 51.0))

    # Coeficientes NORM para n=50, fijados como referencia.
    xp = 0.167 * (1 + 0.3 / 50)
    bp = 0.229 * (1 - 0.2 / 50)
    hp = 1.147 * (1 + 0.5 / 50)
    a2 = 0.5

    esperado = 0.0403 + 0.116 * (
        (a2 - xp) / bp
    ) ** (hp / 0.851)

    assert selector._calc_adc(a2, "NORM") == pytest.approx(
        esperado, rel=1e-12
    )
