"""
Analizador de probabilidades basado en frecuencia histórica
"""

import pandas as pd
from datetime import datetime
from typing import Dict


class ProbabilityAnalyzer:
    """
    Analizador de probabilidades basado en frecuencia histórica.
    Implementa la lógica core del desafío: calcular P(evento climático).
    """

    @staticmethod
    def calcular_probabilidad(
        df: pd.DataFrame,
        metrica: str,
        umbral: float,
        condicion: str
    ) -> Dict:
        """
        Calcula la probabilidad de que se cumpla una condición climática.

        Args:
            df: DataFrame con datos históricos
            metrica: Nombre de la métrica (columna) a evaluar
            umbral: Valor umbral para la condición
            condicion: Tipo de condición ('mayor_que', 'menor_que', 'igual_a')

        Returns:
            Diccionario con resultados del análisis probabilístico
        """
        # Validar que la métrica existe
        if metrica not in df.columns:
            raise ValueError(f"Métrica '{metrica}' no encontrada en el DataFrame")

        # Filtrar valores válidos (no NaN)
        datos_validos = df[metrica].dropna()
        dias_totales = len(datos_validos)

        # Aplicar condición
        if condicion == "mayor_que":
            dias_favorables = (datos_validos > umbral).sum()
            simbolo = ">"
        elif condicion == "menor_que":
            dias_favorables = (datos_validos < umbral).sum()
            simbolo = "<"
        elif condicion == "igual_a":
            dias_favorables = (datos_validos == umbral).sum()
            simbolo = "="
        else:
            raise ValueError(f"Condición '{condicion}' no reconocida")

        # Calcular probabilidad
        probabilidad = (dias_favorables / dias_totales) if dias_totales > 0 else 0

        # Calcular estadísticas adicionales
        percentil_umbral = (datos_validos <= umbral).sum() / dias_totales * 100

        return {
            "probabilidad": round(probabilidad, 4),
            "probabilidad_porcentaje": round(probabilidad * 100, 2),
            "dias_favorables": int(dias_favorables),
            "dias_totales": int(dias_totales),
            "condicion_evaluada": f"{metrica} {simbolo} {umbral}",
            "percentil_umbral": round(percentil_umbral, 2),
            "estadisticas_metrica": {
                "media": round(datos_validos.mean(), 2),
                "mediana": round(datos_validos.median(), 2),
                "desviacion_estandar": round(datos_validos.std(), 2),
                "minimo": round(datos_validos.min(), 2),
                "maximo": round(datos_validos.max(), 2)
            }
        }

    @staticmethod
    def analisis_por_mes(
        df: pd.DataFrame,
        metrica: str,
        umbral: float,
        condicion: str
    ) -> pd.DataFrame:
        """
        Calcula probabilidades desglosadas por mes.

        Args:
            df: DataFrame con datos históricos
            metrica: Métrica a evaluar
            umbral: Valor umbral
            condicion: Tipo de condición

        Returns:
            DataFrame con probabilidades mensuales
        """
        datos_validos = df[[metrica, 'mes']].dropna()

        resultados_mensuales = []
        for mes in range(1, 13):
            datos_mes = datos_validos[datos_validos['mes'] == mes][metrica]

            if condicion == "mayor_que":
                dias_fav = (datos_mes > umbral).sum()
            elif condicion == "menor_que":
                dias_fav = (datos_mes < umbral).sum()
            else:
                dias_fav = (datos_mes == umbral).sum()

            dias_tot = len(datos_mes)
            prob = (dias_fav / dias_tot) if dias_tot > 0 else 0

            resultados_mensuales.append({
                'mes': mes,
                'nombre_mes': datetime(2000, mes, 1).strftime('%B'),
                'probabilidad': round(prob * 100, 2),
                'dias_evaluados': int(dias_tot)
            })

        return pd.DataFrame(resultados_mensuales)
