import numpy as np

from hidroModelSelect import HidroModelSelector


def test_unico_candidato_tras_ks_que_falla_ad_registra_alternativa():
    selector = HidroModelSelector(np.arange(1.0, 53.0))
    selector.results = {
        "Pasa_KS_falla_AD": {
            "aic": 100.0,
            "aicc": 100.3,
            "bic": 104.0,
            "ks_pv": 0.20,
            "ad_c": 5.0,
        },
        "Falla_KS": {
            "aic": 102.0,
            "aicc": 102.3,
            "bic": 106.0,
            "ks_pv": 0.01,
            "ad_c": 0.3,
        },
    }

    elegido = selector.get_best_dist()

    assert elegido.index.tolist() == ["Pasa_KS_falla_AD"]
    assert elegido.iloc[0]["transp"] == "ad_cMax"
    assert (
        selector.results["Pasa_KS_falla_AD"]["transp"]
        == "ad_cMax"
    )
    assert selector.results["Falla_KS"]["transp"] == "Falla KS"
    assert len(selector.results) == 2

def test_unico_candidato_tras_ks_que_pasa_ad_registra_controles():
    selector = HidroModelSelector(np.arange(1.0, 53.0))
    selector.results = {
        "Pasa_KS_y_AD": {
            "aic": 100.0,
            "aicc": 100.3,
            "bic": 104.0,
            "ks_pv": 0.20,
            "ad_c": 0.3,
        },
        "Falla_KS": {
            "aic": 102.0,
            "aicc": 102.3,
            "bic": 106.0,
            "ks_pv": 0.01,
            "ad_c": 0.2,
        },
    }

    elegido = selector.get_best_dist()

    assert elegido.index.tolist() == ["Pasa_KS_y_AD"]
    assert elegido.iloc[0]["transp"] == "ad_cH0"
    assert selector.results["Pasa_KS_y_AD"]["transp"] == "ad_cH0"
    assert selector.results["Falla_KS"]["transp"] == "Falla KS"
    assert len(selector.results) == 2
