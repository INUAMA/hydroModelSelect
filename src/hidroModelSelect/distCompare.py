import numpy as np
from scipy.stats import kstest
from pandas import DataFrame, to_numeric as pdto_numeric
import matplotlib.pyplot as plt

class HidroModelSelector:
    """
    Clase principal para la selección y comparación de modelos de distribución 
    estadística en hidrología (ej. Gumbel, GEV, Pearson3, Normal, Lognormal, SQRT-ETmax).
    Permite ajustar distribuciones, calcular estadísticos de bondad de ajuste (AIC, BIC, ADC) 
    y generar gráficos comparativos.
    """
    
    def __init__(self, data):
        """
        Inicializa el selector de modelos con los datos empíricos observados.
        
        Args:
            data (array-like): Muestra unidimensional, no vacía, de valores
                reales convertibles a float y finitos. Admite ceros y
                valores negativos.

        Raises:
            ValueError: Si la muestra tiene dimensión o tamaño inválidos,
                contiene valores no finitos, complejos o enmascarados,
                o no puede convertirse a un array de tipo float.
        """
        if np.ma.isMaskedArray(data) and np.any(
            np.ma.getmaskarray(data)
        ):
            raise ValueError(
                "La muestra no puede contener observaciones enmascaradas."
            )

        try:
            observations = np.asarray(data)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "La muestra debe contener valores numéricos reales "
                "convertibles a float."
            ) from exc

        if np.iscomplexobj(observations) or (
            observations.dtype.kind == "O"
            and any(
                np.iscomplexobj(value)
                for value in observations.flat
            )
        ):
            raise ValueError(
                "La muestra no puede contener números complejos."
            )

        try:
            observations = np.array(
                observations,
                dtype=float,
                copy=True,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "La muestra debe contener valores numéricos reales "
                "convertibles a float."
            ) from exc

        if observations.ndim != 1:
            raise ValueError("La muestra debe ser unidimensional.")

        if observations.size == 0:
            raise ValueError(
                "La muestra debe contener al menos una observación."
            )
        if not np.all(np.isfinite(observations)):
            raise ValueError(
                "La muestra debe contener solo valores finitos."
            )

        self.obs = observations
        self.obs_sort = np.sort(observations)
        self.n = observations.size
        self.results = {}
        self.fit_errors = {}

        # Coeficientes 'Case 0' (Independientes de la distribución)
        
        self.x0 = 0.0403
        self.b0 = 0.116
        self.h0 = 0.851
        
    @staticmethod
    def pbias_desglosado(obs, sim):
        """
        Calcula el sesgo porcentual (PBIAS) total y lo desglosa en sesgo por 
        omisión (subestimación) y comisión (sobreestimación).

        - PBIAS Total: Mide la tendencia promedio del modelo a sobrestimar o subestimar.
        - PBIAS de Omisión: Cuantifica el sesgo proveniente de eventos donde el modelo subestima (sim < obs).
        - PBIAS de Comisión: Cuantifica el sesgo proveniente de eventos donde el modelo sobrestima (sim > obs).

        Args:
            obs (np.ndarray): Array con los valores observados.
            sim (np.ndarray): Array con los valores simulados por el modelo.

        Returns:
            tuple: Una tupla conteniendo (pbias_total, pbias_omision, pbias_comision).
                   Retorna (np.nan, np.nan, np.nan) si la suma de las observaciones es cero.
        """
        total_obs_volume = np.sum(obs)
        if total_obs_volume == 0:
            return np.nan, np.nan, np.nan

        diff = obs - sim

        # Omisión: El modelo subestima (obs > sim), la diferencia es positiva.
        omission_errors = diff[diff > 0]
        pbias_omision = (np.sum(omission_errors) / total_obs_volume) * 100

        # Comisión: El modelo sobrestima (obs < sim), la diferencia es negativa.
        commission_errors = diff[diff < 0]
        pbias_comision = (np.sum(commission_errors) / total_obs_volume) * 100

        # PBIAS Total: Es la suma neta de los errores.
        pbias_total = (np.sum(diff) / total_obs_volume) * 100

        return pbias_total, pbias_omision, pbias_comision
    
    def _calculate_aic_bic(self, log_lik, k):
        """
        Calcula AIC y BIC.
        AIC = 2k - 2ln(L)
        AICc incluye la corrección para muestras pequeñas.
        Si n <= k + 1, devuelve AICc = +inf.
        BIC = k*ln(n) - 2ln(L)
        
        Entradas:
        - log_lik: logaritmo de la verosimilitud del modelo ajustado
        - k: número de parámetros del modelo
        - n: tamaño de la muestra
        """
        
        aic = 2*k - 2*log_lik
        aicc = aic + (2*k*(k+1)) / (self.n - k - 1) if self.n > (k + 1) else np.inf
        bic = k*np.log(self.n) - 2*log_lik

        return aic, aicc, bic

    def _calculate_anderson_stat(self, logcdf_values, logsf_values):
        """Calcula Anderson–Darling utilizando probabilidades logarítmicas.

        Args:
            logcdf_values (array_like): Log-CDF de las observaciones
                ordenadas de menor a mayor.
            logsf_values (array_like): Log-SF en ese mismo orden.

        Returns:
            float: Estadístico A² de Anderson–Darling.
        """
        logcdf_values = np.asarray(logcdf_values, dtype=float)
        logsf_values = np.asarray(logsf_values, dtype=float)

        i = np.arange(1, self.n + 1)
        suma = np.sum(
            (2 * i - 1)
            * (logcdf_values + logsf_values[::-1])
        )

        return -self.n - suma / self.n

    def _get_laio_coeffs(self, dist_type, shape_param=None):
        """
        Función auxiliar para recuperar coeficientes de las Tablas 3 y 5 de Laio (2004).
        Incluye correcciones asintóticas y de muestra pequeña.
        """
        n = self.n
        sqrt_n = np.sqrt(n)

        if dist_type in ['EV1', 'GUMBEL', 'EV2']: 
            # NOTA: Para EV2 (Frechet), log-transformar datos y usar coeficientes EV1 [7].

            # Tabla 3 (Asintótico) [8]
            xp_inf = 0.169
            bp_inf = 0.229
            hp_inf = 1.141

            # Tabla 5 (Corrección muestra pequeña) [9]
            xp = xp_inf * (1 + 0.1 / n)
            bp = bp_inf * (1 - 0.2 / n)
            hp = hp_inf * (1 + 0.5 / n)

        elif dist_type in ['NORM', 'NORMAL', 'LN', 'LOGNORMAL']:
            # NOTA: Para Lognormal, log-transformar datos y usar coeficientes NORM [7].

            # Tabla 3 [8]
            xp_inf = 0.167
            bp_inf = 0.229
            hp_inf = 1.147

            # Tabla 5 [9]
            xp = xp_inf * (1 + 0.3 / n)
            bp = bp_inf * (1 - 0.2 / n)
            hp = hp_inf * (1 + 0.5 / n)

        elif dist_type == 'GEV':
            if shape_param is None:
                raise ValueError("El parámetro de forma es necesario para GEV.")

            if not np.isfinite(shape_param):
                raise ValueError(
                    "El parámetro de forma GEV debe ser finito."
                )

            if shape_param < -1.0:
                raise ValueError(
                    "El parámetro de forma GEV está fuera del "
                    "dominio de la aproximación de Laio: forma >= -1."
                )

            # Restricción de Laio: si shape > 0.5, usar 0.5 [8]
            k = min(shape_param, 0.5)

            k2 = k**2
            k3 = k**3

            # Tabla 3 (Polinomios de forma) [8]
            xp_inf = 0.147 * (1 + 0.13*k + 0.21*k2 + 0.09*k3)
            bp_inf = 0.189 * (1 + 0.20*k + 0.37*k2 + 0.17*k3)
            hp_inf = 1.186 * (1 - 0.04*k - 0.04*k2 - 0.01*k3)

            # Tabla 5 (Corrección muestra pequeña con forma) [9]
            xp = xp_inf * (1 + 0.9/n - 0.2/sqrt_n)
            bp = bp_inf * (1 - 1.8/n)
            hp = hp_inf * (1 - 0.7/n + 0.2/sqrt_n)

        elif dist_type in ['GAM', 'GAMMA', 'P3', 'LP3']:
            # NOTA: Para Log-Pearson 3, log-transformar y usar GAM [7].
            if shape_param is None:
                raise ValueError("El parámetro de forma es necesario para GAMMA.")

            if not np.isfinite(shape_param) or shape_param <= 0:
                raise ValueError(
                    "El parámetro de forma GAM debe ser finito "
                    "y estrictamente positivo."
                )

            # El paper usa "m" o "gamma" como shape. 
            # Laio denota el parámetro de forma como theta_3.
            # Restricción: si theta_3 < 2, usar 2 [8].
            gamma_param = max(shape_param, 2.0)

            inv_g = 1.0 / gamma_param
            inv_g2 = inv_g**2

            # Tabla 3 [8]
            xp_inf = 0.145 * (1 + 0.17*inv_g + 0.33*inv_g2)
            bp_inf = 0.186 * (1 + 0.34*inv_g + 0.30*inv_g2)
            hp_inf = 1.194 * (1 - 0.04*inv_g - 0.12*inv_g2)

            # Tabla 5: la corrección muestral utiliza la forma original.
            xp = xp_inf * (
                1 + 2.0 / n - 0.3 / sqrt_n
                - 0.4 / (sqrt_n * shape_param)
            )
            bp = bp_inf * (
                1 - 0.5 / n - 0.3 / sqrt_n
                + 0.3 / (sqrt_n * shape_param)
            )
            hp = hp_inf * (
                1 - 1.8 / n + 0.1 / sqrt_n
                + 0.5 / (sqrt_n * shape_param)
            )

        else:
            raise ValueError(f"Distribución {dist_type} no soportada o requiere transformación manual.")

        return xp, bp, hp
    
    def _calc_adc(self, A2, dist_type, shape_param=None):
        """
        Transforma el estadístico A^2 de Anderson-Darling en el criterio ADC (omega)
        estandarizado para selección de modelos, según Laio (2004, 2009).

        Entradas:
        - A2: Estadístico Anderson-Darling crudo calculado previamente.
        - dist_type: Tipo de distribución ('EV1', 'GEV', 'NORM', 'GAM').
          * Para Lognormal usar 'NORM' con datos log-transformados.
          * Para Pearson III usar 'GAM' con datos transformados.
        - Tamaño muestral: se obtiene de self.n.
        - shape_param: Parámetro de forma (solo para GEV y GAM).
                       Para GEV es theta3 de Laio (2004), coincidente
                       con c de scipy.stats.genextreme, sin invertir el signo.
                       Para Gamma es el parámetro de forma alpha.

        Salida:
        - w (ADC): Estadístico transformado comparable entre distribuciones.
        """
        # Obtener coeficientes específicos de la distribución (xp, bp, hp)
        # Se obtienen de las Tablas 3 (asintóticos) y 5 (corrección muestra pequeña) de Laio (2004).
        xp, bp, hp = self._get_laio_coeffs(dist_type, shape_param)

        # Ecuación de transformación (Laio 2004 Eq 11; Laio 2009 Eq 4)
        # Se divide en dos tramos para mejorar precisión en la cola inferior [5], [1].

        if A2 >= 1.2 * xp:
            # Tramo principal
            term = (A2 - xp) / bp
            # Evitar bases negativas por seguridad numérica (aunque A2 >= 1.2xp lo previene)
            term = max(term, 1e-10) 
            w = self.x0 + self.b0 * (term ** (hp / self.h0))
        else:
            # Tramo para valores pequeños de A2 (corrección de cola inferior)
            term1 = ((0.2 * xp) / bp) ** (hp / self.h0)
            term2 = (A2 - 0.2 * xp) / xp
            w = (self.x0 + self.b0 * term1) * term2

        return w
    
    def _ad_corr(self, ad_stat, n):
        """
        Aplica la corrección de Stephens para muestras finitas.
        Válido para distribuciones de valores extremos (Gumbel, GEV, Pearson III).
        """
        # La fórmula estándar de Stephens usa 0.75
        return ad_stat * (1 + (0.75 / n) + (2.25 / (n ** 2)))

    @staticmethod
    def _validate_fitted_params(params):
        """Comprueba parámetros finitos y una escala positiva."""
        if not np.all(np.isfinite(params)):
            raise ValueError(
                "Los parámetros ajustados deben ser finitos."
            )

        if params[-1] <= 0:
            raise ValueError(
                "La escala ajustada debe ser estrictamente positiva."
            )

    def _update_aicc_deltas(self):
        """Calcula diferencias solo entre valores finitos de AICc."""
        aicc_finitos = [
            resultado["aicc"]
            for resultado in self.results.values()
            if np.isfinite(resultado["aicc"])
        ]

        min_aicc = min(aicc_finitos) if aicc_finitos else None

        for resultado in self.results.values():
            if (
                min_aicc is None
                or not np.isfinite(resultado["aicc"])
            ):
                resultado["d_aicc"] = np.nan
            else:
                resultado["d_aicc"] = (
                    resultado["aicc"] - min_aicc
                )

    def fit_distribution(self, name, dist_obj, is_custom=False, custom_type="mle", **kwargs):
        """Ajusta una distribución y almacena sus estadísticas.

        Los resultados se guardan en self.results. Cada ajuste exitoso
        registra el método solicitado y normalizado en fit_method.

        Los fallos capturados se registran en self.fit_errors y eliminan
        cualquier resultado anterior del candidato. Un ajuste exitoso
        elimina su error previo.

        Args:
            name (str): Nombre visible del candidato.
            dist_obj (object): Objeto de distribución.
            is_custom (bool): Activa el ajuste personalizado de SQRT-ETmax.
            custom_type (str): Estimador personalizado: "mle" por defecto,
                "mel" como alias de "mle", o "lmoments".
                Se valida únicamente cuando is_custom=True.
            **kwargs: Argumentos adicionales para fit en la ruta genérica.
                method admite "MLE" y "MM", sin distinguir mayúsculas
                y minúsculas. Por defecto se utiliza "mle".
                Las opciones de fijación con valor None no descuentan
                parámetros libres. Los valores realmente fijados,
                incluido cero, sí se descuentan.

        Raises:
            ValueError: Si custom_type no es válido en la ruta personalizada
                o method no es válido en la ruta genérica. La validación
                se realiza antes del ajuste y conserva el estado previo.
        """

        if is_custom:
            if not isinstance(custom_type, str) or custom_type not in (
                "mle", "mel", "lmoments",
            ):
                raise ValueError(
                    "custom_type debe ser una cadena: "
                    "'mle', 'mel' o 'lmoments'."
                )

            if custom_type == "mel":
                custom_type = "mle"

        else:
            method = kwargs.get("method", "mle")

            if not isinstance(method, str) or method.lower() not in (
                "mle", "mm",
            ):
                raise ValueError(
                    "method debe ser una cadena: 'mle' o 'mm'."
                )

            kwargs["method"] = method.lower()

        try:
            if is_custom:
                if custom_type == "mle":
                    # Ajuste por máxima verosimilitud para SQRT-ETmax.
                    # fit_custom retorna (k, alpha), scipy espera (k, loc, scale)
                    k_fit, alpha_fit = dist_obj.fit_custom(self.obs_sort)
                    params = (k_fit, 0, 1.0/alpha_fit)
                    self._validate_fitted_params(params)
                    k_params = 2
                    cdf_vals = dist_obj.cdf(self.obs_sort, *params)
                    dist_type_laio = None # No soportado para ADC
                elif custom_type == "lmoments":
                    name = f"{name}_Lmom"
                    # Ajuste específico para SQRT-ETmax L-moment.
                     # fit_lmoments retorna (k, alpha); SciPy espera (k, loc, scale).
                    k_fit, alpha_fit = dist_obj.fit_lmoments(self.obs_sort)
                    params = (k_fit, 0, 1.0/alpha_fit)
                    self._validate_fitted_params(params)
                    k_params = 2
                    cdf_vals = dist_obj.cdf(self.obs_sort, *params)
                    dist_type_laio = None # No soportado para ADC
                
            else:
                family = getattr(dist_obj, "name", None)

                # Localización fija en cero por defecto para estas familias.
                if (
                    family in ("lognorm", "sqrt_etmax")
                    and "floc" not in kwargs
                ):
                    kwargs["floc"] = 0
                    
                # Ajuste estándar Scipy con posibles parámetros fijados
                params = dist_obj.fit(self.obs_sort, **kwargs)
                self._validate_fitted_params(params)
                
                # None no representa un parámetro fijado.
                n_fixed = sum(
                    1
                    for key, value in kwargs.items()
                    if key.startswith("f") and value is not None
                )
                k_params = len(params) - n_fixed
                log_pdf = dist_obj.logpdf(self.obs_sort, *params)
                cdf_vals = dist_obj.cdf(self.obs_sort, *params)
                
                # Mapeo a tipos de Laio
                dist_type_laio = {
                    "gumbel_r": "EV1",
                    "norm": "NORM",
                    "lognorm": "NORM",
                    "genextreme": "GEV",
                    "pearson3": "GAM",
                }.get(family)

            if not np.all(np.isfinite(cdf_vals)):
                raise ValueError(
                    "La CDF debe contener solo valores finitos."
                )

            if np.any((cdf_vals < 0) | (cdf_vals > 1)):
                raise ValueError(
                    "Los valores de la CDF deben estar entre 0 y 1."
                )

            # 1. Criterios de Información
            if is_custom:
                log_lik = dist_obj.log_likelihood(
                    self.obs_sort,
                    k=k_fit,
                    alpha=alpha_fit,
                )
            else:
                log_lik = np.sum(log_pdf)

            if not np.isfinite(log_lik):
                raise ValueError(
                    "La log-verosimilitud debe ser finita."
                )

            aic, aicc, bic = self._calculate_aic_bic(log_lik, k_params)

            if self.n > k_params + 1:
                aicc_valido = np.isfinite(aicc)
            else:
                # Convención actual para AICc no definido.
                aicc_valido = np.isposinf(aicc)

            if (
                not np.all(np.isfinite([aic, bic]))
                or not aicc_valido
            ):
                raise ValueError(
                    "Los criterios de información contienen "
                    "valores inválidos."
                )
            
            # 2. Anderson-Darling (A2 y ADC)
            a2 = self._calculate_anderson_stat(
                dist_obj.logcdf(self.obs_sort, *params),
                dist_obj.logsf(self.obs_sort, *params),
            )
            adc = np.nan

            adc_reason = (
                None if dist_type_laio is not None
                else "Familia sin corrección ADC implementada."
            )
            
            if dist_type_laio:
                shape_val = None

                if family == "genextreme":
                    c, loc, scale = params
                    # Laio (2004), tabla 1: theta3 tiene el mismo signo que c de SciPy.
                    shape_val = c

                elif family == "pearson3":
                    skew, loc, scale = params
                    shape_val = (
                        (2.0 / skew) ** 2
                        if abs(skew) > 1e-4
                        else 1000.0
                    )

                if kwargs["method"] != "mle":
                    adc_reason = (
                        "ADC no disponible para el estimador "
                        f"'{kwargs['method']}': esta corrección "
                        "requiere un estimador compatible con Laio."
                    )
                elif family in (
                    "norm", "gumbel_r", "genextreme", "pearson3",
                ) and any(
                    key.startswith("f") and value is not None
                    for key, value in kwargs.items()
                ):
                    adc_reason = (
                        "ADC no disponible: la corrección "
                        "implementada para esta familia requiere "
                        "estimar sus parámetros, sin parámetros fijados."
                    )
                elif (
                    family == "lognorm"
                    and kwargs.get("floc") is None
                ):
                    adc_reason = (
                        "ADC no disponible: la corrección Lognormal "
                        "implementada requiere una localización "
                        "fijada, no estimada desde la muestra."
                    )
                elif family == "lognorm" and any(
                    key.startswith("f")
                    and key != "floc"
                    and value is not None
                    for key, value in kwargs.items()
                ):
                    adc_reason = (
                        "ADC no disponible: la corrección Lognormal "
                        "requiere estimar forma y escala, sin "
                        "parámetros fijados adicionales a la localización."
                    )
                elif dist_type_laio == "GEV" and shape_val < -1.0:
                    adc_reason = (
                        "Forma GEV fuera del dominio de la "
                        "aproximación de Laio: forma >= -1."
                    )
                elif dist_type_laio == "GEV" and shape_val >= 0.5:
                    adc_reason = (
                        "ADC no disponible: el estimador MLE genérico "
                        "no acredita las condiciones de Laio para "
                        "forma GEV >= 0.5."
                    )
                elif dist_type_laio == "GAM" and shape_val <= 2.0:
                    adc_reason = (
                        "ADC no disponible: el estimador MLE genérico "
                        "no acredita las condiciones de Laio para "
                        "forma GAM <= 2."
                    )
                elif dist_type_laio in ("NORM", "EV1") and self.n < 10:
                    adc_reason = (
                        "ADC no disponible: el tamaño muestral "
                        f"n={self.n} es inferior al mínimo de 10 "
                        "adoptado para esta familia."
                    )
                elif dist_type_laio in ("GEV", "GAM") and self.n < 20:
                    adc_reason = (
                        "ADC no disponible: el tamaño muestral "
                        f"n={self.n} es inferior al mínimo de 20 "
                        "adoptado para esta familia."
                    )
                else:
                    adc = self._calc_adc(
                        a2, dist_type_laio, shape_val,
                    )

            # 3. Kolmogorov-Smirnov (Útil para SQRT-ETmax)
            ks_stat, ks_pv = kstest(self.obs_sort, lambda x: dist_obj.cdf(x, *params))
            
            # 4. AD corregido ( D'Agostino & Stephens (1986))
            
            ad_c = self._ad_corr(a2,self.n)

            estadisticos = [a2, ad_c, ks_stat, ks_pv]

            if adc_reason is None:
                estadisticos.append(adc)

            if not np.all(np.isfinite(estadisticos)):
                raise ValueError(
                    "Los estadísticos de bondad de ajuste "
                    "deben ser finitos."
                )

            self.results[name] = {
                'aic': aic, 'aicc': aicc, 'bic': bic,
                'a2': a2, 'adc': adc, 'ad_c': ad_c,
                'ks': ks_stat, 'ks_pv': ks_pv,
                'params': params,
                'k_params': k_params,
                'adc_reason': adc_reason
            }

            if is_custom:
                self.results[name]["fit_method"] = custom_type
            else:
                self.results[name]["fit_method"] = kwargs["method"]

            self._update_aicc_deltas()
            self.fit_errors.pop(name, None)
            
        except Exception as e:
            self.results.pop(name, None)
            self.fit_errors[name] = {
                "error_type": type(e).__name__,
                "message": str(e),
            }
            self._update_aicc_deltas()
            print(f"Error ajustando {name}: {e}")

    def get_ranking_dataframe(self):
        """Devuelve el ranking por AICc, vacío si no hay ajustes."""
        df = DataFrame(self.results).T

        if df.empty:
            return df

        return df.sort_values("aicc")
    
    def get_ad_critical_value(self, n, alpha=0.05):
        """
        Calcula el valor crítico de Anderson-Darling ajustado por tamaño muestral

        Parámetros:
        n: tamaño de la muestra
        alpha: nivel de significancia (0.05 para 95%, 0.01 para 99%, etc.)
        """
        # Valores críticos teóricos para distribución normal
        crit_values = {
            0.10: 0.656,  # 90%
            0.05: 0.787,  # 95%
            0.025: 0.918, # 97.5%
            0.01: 1.092   # 99%
        }

        # Obtener el valor teórico
        ad_teorico = crit_values.get(alpha, 0.787)  # default 95%

        # Ajuste para muestras pequeñas
        if n <= 100:  # El ajuste es relevante para n < 100
            ad_critico = ad_teorico * (1 + 0.6 / n)
        else:
            ad_critico = ad_teorico  # Para muestras grandes, el ajuste es despreciable

        return ad_critico
    
    def get_best_dist(self, criterion="aicc"):
        """
        Selecciona un modelo mediante filtros jerárquicos y un criterio común.

        Primero excluye de la selección los candidatos sin un valor numérico
        y finito del criterio solicitado. Después aplica los filtros KS y
        AD corregido, conservando sus reglas de selección y alternativas.

        Cuando varios candidatos superan ambos filtros, conserva aquellos
        con una diferencia <= 2 respecto al menor valor del criterio.
        Entre ellos selecciona el de menor AD corregido.

        Args:
            criterion (str): Criterio común para todos los candidatos.
                Admite "aic", "aicc" y "bic". Por defecto, "aicc".

        Returns:
            pandas.DataFrame: Fila del modelo seleccionado, con el criterio
                en "criterion", su valor en "metri" y la trazabilidad
                en "transp".

        Raises:
            ValueError: Si criterion no es una opción admitida.
            RuntimeError: Si no hay ajustes disponibles o ninguno tiene
                un valor finito del criterio solicitado.

        Actualiza "criterion" y "transp" en self.results para describir
        la última selección. Conserva los ajustes y sus estadísticas,
        incluidos los candidatos excluidos de esa selección.
        """
        if not isinstance(criterion, str) or criterion not in (
            "aic", "aicc", "bic",
        ):
            raise ValueError(
                "criterion debe ser una cadena: "
                "'aic', 'aicc' o 'bic'."
            )

        df = self.get_ranking_dataframe()

        if df.empty:
            raise RuntimeError(
                "No hay ajustes disponibles para seleccionar."
            )

        df["criterion"] = criterion

        for nombre in df.index:
            if nombre in self.results:
                self.results[nombre]["criterion"] = criterion

        # Criterio común para todos los candidatos.
        df["metri"] = pdto_numeric(
            df[criterion],
            errors="coerce",
        )
        df["metri"] = pdto_numeric(
            df[criterion],
            errors="coerce",
        )

        df["ad_c"] = pdto_numeric(
            df["ad_c"],
            errors="raise",
        )
        disponibles = np.isfinite(df["metri"])

        for nombre in df.index[~disponibles]:
            if nombre in self.results:
                self.results[nombre]["transp"] = (
                    "Criterio no disponible"
                )

        df = df.loc[disponibles].copy()

        if df.empty:
            raise RuntimeError(
                "No hay ajustes con valores finitos para "
                f"el criterio '{criterion}'."
            )
            
        # Marcamos el estado por defecto de todas como rechazadas en el primer filtro
        df['transp'] = 'Falla KS'
        
        mejor_idx = None
        
        # Criterio 1: ks_pv >= 0.05
        mask_ks = df['ks_pv'] >= 0.05
        df.loc[mask_ks, 'transp'] = 'Falla AD' # Las que pasan KS, caen en AD por defecto
        validas = df[mask_ks].copy()
        
        if validas.empty:
            mejor_idx = df['ks_pv'].idxmax()
            df.loc[mejor_idx, 'transp'] = 'pv_max'
        else:
            # Criterio 2: ad_c <= ad_critico (Test de Anderson-Darling al 95%)
            # Evaluar AD aunque solo un candidato haya superado KS.
            ad_critico = self.get_ad_critical_value(self.n)
            mask_ad = mask_ks & (df['ad_c'] <= ad_critico)
            df.loc[mask_ad, 'transp'] = 'Falla Optimo' # Las que pasan AD, caen en Óptimo por defecto
            
            validas2 = df[mask_ad].copy()
            
            if validas2.empty:
                # Si ninguna cumple el valor crítico, tomar la de menor ad_c entre las que pasaron KS
                mejor_idx = validas['ad_c'].idxmin()
                df.loc[mejor_idx, 'transp'] = 'ad_cMax'
            elif len(validas2) == 1:
                mejor_idx = validas2.index[0]
                df.loc[mejor_idx, 'transp'] = 'ad_cH0'
            else:
                # Criterio 3: criterio común solicitado y desempate por AD.
                try:
                    min_val = validas2['metri'].min()
                    mask_optima = mask_ad & (df['metri'] - min_val <= 2.0)
                    df.loc[mask_optima, 'transp'] = 'Desempate AD' # Óptimas pero no ganadoras
                    
                    validas3 = df[mask_optima].copy()
                    validas3['ad_c'] = pdto_numeric(validas3['ad_c'], errors='coerce')

                    mejor_idx = validas3['ad_c'].idxmin()
                    df.loc[mejor_idx, 'transp'] = 'optima_ci'
                except Exception as e:
                    print(f"Aviso: Fallo en Criterio 3 ({e}). Aplicando fallback.")
                    mejor_idx = validas2['ad_c'].idxmin()
                    df.loc[mejor_idx, 'transp'] = 'optima_ci_fallback'
                    
        # Sincronizar la trazabilidad con self.results
        for idx, row in df.iterrows():
            if idx in self.results:
                self.results[idx]['transp'] = row['transp']
                
        return df.loc[[mejor_idx]]
            
    def plotCCAcum(self, dist_obj={}, dist_n=1, path_result=None, order_stats='aicc'):
        """
        Genera un gráfico comparativo de las curvas de probabilidad acumulada (CDF).
        
        Args:
            dist_obj (dict): Diccionario con los objetos de distribución de scipy.stats.
            dist_n (int): Número de distribuciones del top a graficar.
            path_result (str, opcional): Ruta donde guardar la figura. Si es None, se muestra en pantalla.
            order_stats (str): Criterio estadístico para ordenar los modelos (por defecto 'aicc').
        """
        y_obs = np.arange(1, self.n +1) / self.n
        x_t = np.linspace(min(self.obs_sort), max(self.obs_sort), 100)
        lines = DataFrame(self.results).T
        dist_max = len(lines.iloc[0])
        if dist_n > dist_max : 
            dist_n = dist_max
            print(f"Número máximo de distribuciones calculadas: {dist_max}")
        lines = lines.sort_values(order_stats).iloc[0:dist_n]
        y_t = []
        dist = None
        col = None
        COLOR = {
            'Gumbel': '#E41A1C',        # Rojo - distribución extrema común
            'GEV': '#377EB8',           # Azul - generalización de Gumbel
            'Pearson3': '#4DAF4A',      # Verde - asimétrica
            'Normal': '#984EA3',        # Púrpura - fundamental
            'Log_Normal': '#FF7F00',    # Naranja - logarítmica
            'SQRT-ETmax': '#A65628',    # Marrón - transformación
            'SQRT-ETmax_Lmom': '#999999' # Gris - variante con L-momentos
        }
        plt.step(self.obs_sort, y_obs, where='post', label='Observado (Empírica)', color='blue')
        for eti, fila in lines.iterrows():
            dist = dist_obj[eti]
            y_t = dist.cdf(x_t, *fila.params)
            col = COLOR[eti]
            plt.plot(x_t, y_t, label=f'Predicho ({eti})', color=col, linewidth=2)
        plt.xlabel('Precipitación (mm)')
        plt.ylabel('Probabilidad Acumulada F(x)')
        plt.title('Comparación de Curvas Acumuladas (S-Curve)')
        plt.legend()
        plt.grid(True)
        if path_result is not None:
            plt.savefig(path_result)
        else:
            plt.show()
            
    def plot_qq(self, dist_obj={}, dist_n=1, path_result=None, order_stats='aicc'):
        """
        Genera un gráfico Q-Q comparando los datos observados con los cuantiles teóricos.
        """
        # 1. Probabilidades empíricas (Posición de Gringorten)
        # p = (i - 0.44) / (n + 0.12)
        i = np.arange(1, self.n + 1)
        p = (i - 0.44) / (self.n + 0.12)
        
        plt.figure(figsize=(8, 8))
        
        lines = DataFrame(self.results).T
        dist_max = len(lines)
        if dist_n > dist_max: 
            dist_n = dist_max
        
        lines = lines.sort_values(order_stats).iloc[0:dist_n]
        
        COLOR = {
            'Gumbel': '#E41A1C', 'GEV': '#377EB8', 'Pearson3': '#4DAF4A',
            'Normal': '#984EA3', 'Log_Normal': '#FF7F00',
            'SQRT-ETmax': '#A65628', 'SQRT-ETmax_Lmom': '#999999'
        }
        
        for name, row in lines.iterrows():
            if name in dist_obj:
                dist = dist_obj[name]
                params = row['params']
                try:
                    # Calcular cuantiles teóricos
                    theo = dist.ppf(p, *params)
                    col = COLOR.get(name, 'black')
                    plt.scatter(theo, self.obs_sort, s=20, alpha=0.7, label=f"{name}", color=col, edgecolors='none')
                except Exception as e:
                    print(f"Error graficando {name}: {e}")
        
        # Línea 1:1
        min_val = self.obs_sort.min()
        max_val = self.obs_sort.max()
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='1:1')
        
        plt.xlabel('Teóricos (Simulados)')
        plt.ylabel('Observados')
        plt.title('Gráfico Q-Q')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        if path_result is not None:
            plt.savefig(path_result)
        else:
            plt.show()
            
    def plot_return_levels(self, dist_obj={}, dist_n=1, path_result=None, order_stats='aicc'):
        """
        Genera un gráfico de Niveles de Retorno (Return Level Plot).
        Eje X: Periodo de Retorno (T) en escala logarítmica.
        Eje Y: Precipitación (Cuantiles).
        """
        # 1. Datos Observados (Posición de Gringorten)
        # p = (i - 0.44) / (n + 0.12)
        i = np.arange(1, self.n + 1)
        p_emp = (i - 0.44) / (self.n + 0.12)
        T_emp = 1.0 / (1.0 - p_emp)
        
        plt.figure(figsize=(10, 6))
        
        # Graficar Observados
        plt.scatter(T_emp, self.obs_sort, color='black', marker='o', s=25, label='Observado', zorder=3)
        
        # 2. Modelos Teóricos
        # Generar eje T para las líneas (suave)
        # Desde T=1.01 hasta un poco más del máximo observado
        T_max = max(T_emp.max() * 2, 200) 
        T_axis = np.logspace(np.log10(1.01), np.log10(T_max), 200)
        p_axis = 1.0 - (1.0 / T_axis)
        
        lines = DataFrame(self.results).T
        dist_max = len(lines)
        if dist_n > dist_max: 
            dist_n = dist_max
        
        lines = lines.sort_values(order_stats).iloc[0:dist_n]
        
        COLOR = {
            'Gumbel': '#E41A1C', 'GEV': '#377EB8', 'Pearson3': '#4DAF4A',
            'Normal': '#984EA3', 'Log_Normal': '#FF7F00',
            'SQRT-ETmax': '#A65628', 'SQRT-ETmax_Lmom': '#999999'
        }
        
        for name, row in lines.iterrows():
            if name in dist_obj:
                dist = dist_obj[name]
                params = row['params']
                try:
                    # Calcular cuantiles teóricos para las probabilidades del eje
                    quantiles = dist.ppf(p_axis, *params)
                    col = COLOR.get(name, 'gray')
                    plt.plot(T_axis, quantiles, label=f"{name}", color=col, linewidth=2)
                except Exception as e:
                    print(f"Error graficando {name}: {e}")
        
        plt.xscale('log')
        plt.xlabel('Periodo de Retorno (años)')
        plt.ylabel('Precipitación (mm)')
        plt.title('Gráfico de Niveles de Retorno')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        # Ticks personalizados para T para mejor lectura
        import matplotlib.ticker as ticker
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks([1.5, 2, 5, 10, 25, 50, 100, 200, 500])
        
        plt.legend()
        
        if path_result is not None:
            plt.savefig(path_result)
        else:
            plt.show()