import numpy as np
from scipy.stats import kstest
from pandas import DataFrame
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
            data (array-like): Serie de datos hidrológicos observados (ej. precipitaciones máximas anuales).
        """
        self.data = np.sort(np.array(data, dtype=float))
        self.n = len(self.data)
        self.results = {}
        
        # Coeficientes 'Case 0' (Independientes de la distribución)
        
        self.x0 = 0.0403
        self.b0 = 0.116
        self.h0 = 0.851

    def _calculate_aic_bic(self, log_lik, k):
        """
        Calcula AIC y BIC.
        AIC = 2k - 2ln(L)
        AICc (corregido) se usa si n/k < 40
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

    def _calculate_anderson_stat(self, cdf_values):
        """
        Calcula el estadístico A^2 de Anderson-Darling para cualquier distribución.
        Fórmula: A^2 = -n - (1/n) * sum( (2i-1) * [ln(F(Yi)) + ln(1-F(Yn+1-i))] )
        Donde Yi son los datos ordenados.
        """

        # Calcular CDF para los datos ordenados
        
        cdf_v = np.clip(cdf_values, 1e-10, 1 - 1e-10)

        # Calcular términos de la sumatoria
        i = np.arange(1, self.n + 1)
        term1 = np.log(cdf_v)
        term2 = np.log(1 - cdf_v[::-1]) # Invertir orden para Yn+1-i

        S = np.sum((2*i - 1) * (term1 + term2))
        A2 = -self.n - (1/self.n) * S

        return A2

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

            # Restricción de Laio: si shape > 0.5, usar 0.5 [8]
            k = min(shape_param, 0.5)
            # Se asume rango válido [-0.5, 0.5] aprox.

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

            # Tabla 5 [9]
            # Nota: Laio usa el shape estimado en la corrección
            xp = xp_inf * (1 + 2.0/n - 0.3/sqrt_n - 0.4/(sqrt_n * gamma_param))
            bp = bp_inf * (1 - 0.5/n - 0.3/sqrt_n + 0.3/(sqrt_n * gamma_param))
            hp = hp_inf * (1 - 1.8/n + 0.1/sqrt_n + 0.5/(sqrt_n * gamma_param))

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
        - n: Tamaño de la muestra.
        - shape_param: Parámetro de forma (solo para GEV y GAM). 
                       Para GEV es usualmente 'k' o 'xi'.
                       Para Gamma es 'alpha' o 'k'.

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
            w = self.x0 + self.b0 * term1 * term2

        return w
    
    def fit_distribution(self, name, dist_obj, is_custom=False, custom_type="mel"):
        """
        Ajusta una distribución y calcula sus estadísticas.
        
        :param name: Nombre de la distribución (ej. 'Gumbel', 'SQRT_ETmax')
        :param dist_obj: Objeto de scipy.stats o instancia personalizada (SQRT)
        :param is_custom: True si es SQRT-ETmax (usa fit_custom) [13]
        """
        try:
            if is_custom:
                if custom_type == 'mel':
                    # Ajuste específico para SQRT-ETmax MEL [13]
                    # fit_custom retorna (k, alpha), scipy espera (k, loc, scale)
                    k_fit, alpha_fit = dist_obj.fit_custom(self.data)
                    params = (k_fit, 0, 1.0/alpha_fit)
                    k_params = 2
                    # Calcular log-pdf usando el método interno
                    log_pdf = dist_obj.logpdf(self.data, *params)
                    cdf_vals = dist_obj.cdf(self.data, *params)
                    dist_type_laio = None # No soportado para ADC
                else: 
                    name = f"{name}_Lmom"
                    # Ajuste específico para SQRT-ETmax L-moment [13]
                    # fit_custom retorna (k, alpha), scipy espera (k, loc, scale)
                    k_fit, alpha_fit = dist_obj.fit_lmoments(self.data)
                    params = (k_fit, 0, 1.0/alpha_fit)
                    k_params = 2
                    # Calcular log-pdf usando el método interno
                    log_pdf = dist_obj.logpdf(self.data, *params)
                    cdf_vals = dist_obj.cdf(self.data, *params)
                    dist_type_laio = None # No soportado para ADC
                
            else:
                # Ajuste estándar Scipy
                params = dist_obj.fit(self.data)
                k_params = len(params)
                log_pdf = dist_obj.logpdf(self.data, *params)
                cdf_vals = dist_obj.cdf(self.data, *params)
                
                # Mapeo a tipos de Laio
                if name == 'Gumbel': dist_type_laio = 'EV1'
                elif name == 'Normal': dist_type_laio = 'NORM'
                elif name == 'Log_Normal': dist_type_laio = 'NORM' # Requiere log-data previo si se hace manual
                elif name == 'GEV': dist_type_laio = 'GEV'
                elif name == 'Pearson3': dist_type_laio = 'GAM'
                else: dist_type_laio = None

            # 1. Criterios de Información
            log_lik = np.sum(log_pdf)
            aic, aicc, bic = self._calculate_aic_bic(log_lik, k_params)
            
            # 2. Anderson-Darling (A2 y ADC)
            a2 = self._calculate_anderson_stat(cdf_vals)
            adc = np.nan
            
            if dist_type_laio:
                shape_val = None
                # Lógica de forma para GEV y Pearson3 [14], [15]
                if name == 'GEV':
                    # Scipy c = -k de Laio.
                    c, loc, scale = params 
                    shape_val = -c
                elif name == 'Pearson3':
                    # Skew a Shape (alpha) para Gamma
                    skew, loc, scale = params
                    shape_val = (2.0/skew)**2 if abs(skew) > 1e-4 else 1000.0
                
                adc = self._calc_adc(a2, dist_type_laio, shape_val)

            # 3. Kolmogorov-Smirnov (Útil para SQRT-ETmax)
            ks_stat, ks_pv = kstest(self.data, lambda x: dist_obj.cdf(x, *params))

            self.results[name] = {
                'aic': aic, 'aicc': aicc, 'bic': bic,
                'a2': a2, 'adc': adc, 'ks': ks_stat, 'ks_pv': ks_pv,
                'params': params
            }
            min_aicc = min(map(lambda d: d['aicc'], self.results.values()))
            for model in self.results: self.results[model]['d_aicc'] = self.results[model]['aicc'] - min_aicc
            del min_aicc
            
        except Exception as e:
            print(f"Error ajustando {name}: {e}")

    def get_ranking_dataframe(self):
        """Retorna un DataFrame ordenado por AICc."""
        df = DataFrame(self.results).T
        return df.sort_values('aicc')
    
    @staticmethod
    def get_best_dist(df, n):
        """
        Selecciona la mejor distribución basada en criterios secuenciales
        SIEMPRE retorna un DataFrame NO VACÍO
        """
        # Criterio 1: ks_pv >= 0.05
        validas = df[df['ks_pv'] >= 0.05].copy()
        print(f"Tras ks_pv: {len(validas)} distribuciones")

        if validas.empty | len(validas) == 1:
            # Si ninguna cumple, tomar la de mayor ks_pv
            max_ks = df['ks_pv'].max()
            print(f"  Ninguna cumple ks_pv, tomando max: {max_ks}")
            return df[df['ks_pv'] == max_ks]


        # Criterio 2: a2 <= 0.5
        validas2 = validas[validas['a2'] <= 0.5].copy()
        print(f"Tras a2: {len(validas2)} distribuciones")

        if validas2.empty | len(validas2) == 1:
            # Si ninguna cumple a2, tomar la de menor a2
            min_a2 = validas['a2'].min()
            print(f"  Ninguna cumple a2, tomando min: {min_a2}")
            return validas[validas['a2'] == min_a2]

        # Criterio 3: BIC o AIC según n
        try:
            if n > 40:
                min_val = validas2['bic'].min()
                print(f"Min BIC: {min_val}")
                mascara = (validas2['bic'] - min_val <= 1.0)
                validas3 = validas2[mascara].copy()
            else:
                min_val = validas2['aic'].min()
                print(f"Min AIC: {min_val}")
                mascara = (validas2['aic'] - min_val <= 1.0)
                validas3 = validas2[mascara].copy()
        except Exception as e:
            print(f"Error en criterio 3: {e}")
            return validas[validas['a2'] == min_a2]

        min_a2 = validas3['a2'].min()

        return validas3[validas3['a2'] == min_a2]

    #def best_dist(self):
    #    # criterios
    #    df = self.get_ranking_dataframe()
    #    # Comenzamos por el principio de parsimonia con ajustes equivalentes
    #    # Esta condición siempre se cumple devolverá 1 o más
    #    equi = df[df['d_aicc'] <= 2.0]
    #    
    #    if len(equi['d_aicc']) == 1:
    #        return equi
    #    min_a2 = equi['a2'].min()
    #    
    #    best_tail = equi[equi['a2'] == min_a2]
    #    
    #    if len(best_tail) == 1:
    #        return best_tail
    #    
    #    return best_tail[best_tail['ks_pv'] == best_tail['ks_pv'].max()]   
            
            
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
        x_t = np.linspace(min(self.data), max(self.data), 100)
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
        plt.step(self.data, y_obs, where='post', label='Observado (Empírica)', color='blue')
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
        
        # 2. Datos observados (ya ordenados en __init__)
        obs = self.data
        
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
                    plt.scatter(theo, obs, s=20, alpha=0.7, label=f"{name}", color=col, edgecolors='none')
                except Exception as e:
                    print(f"Error graficando {name}: {e}")
        
        # Línea 1:1
        min_val = obs.min()
        max_val = obs.max()
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
        plt.scatter(T_emp, self.data, color='black', marker='o', s=25, label='Observado', zorder=3)
        
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