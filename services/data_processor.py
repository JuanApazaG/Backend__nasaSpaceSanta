"""
Procesador de datos climáticos
Convierte respuestas JSON en DataFrames estructurados
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List


class DataProcessor:
    """
    Procesador de datos climáticos. Convierte respuestas JSON de la API
    en un DataFrame estructurado y listo para análisis.
    """

    @staticmethod
    def crear_dataframe_maestro(respuestas_api: List[Dict]) -> pd.DataFrame:
        """
        Convierte múltiples respuestas JSON en un DataFrame unificado.

        Args:
            respuestas_api: Lista de respuestas JSON de la API

        Returns:
            DataFrame con índice temporal y columnas de métricas climáticas
        """
        registros = []

        for respuesta in respuestas_api:
            try:
                parametros = respuesta['properties']['parameter']

                # Extraer todas las fechas disponibles (de cualquier parámetro)
                fechas = list(parametros['T2M'].keys())

                # Iterar sobre cada fecha
                for fecha_str in fechas:
                    registro = {
                        'fecha': datetime.strptime(fecha_str, "%Y%m%d"),
                        'T2M_MAX': parametros.get('T2M_MAX', {}).get(fecha_str, np.nan),
                        'T2M_MIN': parametros.get('T2M_MIN', {}).get(fecha_str, np.nan),
                        'T2M': parametros.get('T2M', {}).get(fecha_str, np.nan),
                        'WS10M_MAX': parametros.get('WS10M_MAX', {}).get(fecha_str, np.nan),
                        'PRECTOTCORR': parametros.get('PRECTOTCORR', {}).get(fecha_str, np.nan),
                        'ALLSKY_SFC_SW_DWN': parametros.get('ALLSKY_SFC_SW_DWN', {}).get(fecha_str, np.nan)
                    }
                    registros.append(registro)

            except Exception as e:
                print(f"⚠️ Error procesando respuesta: {e}")
                continue

        # Crear DataFrame
        df = pd.DataFrame(registros)

        # Establecer índice temporal
        df.set_index('fecha', inplace=True)
        df.sort_index(inplace=True)

        # Agregar columnas derivadas útiles
        df['mes'] = df.index.month
        df['dia'] = df.index.day
        df['año'] = df.index.year
        df['dia_año'] = df.index.dayofyear

        return df

    @staticmethod
    def generar_estadisticas(df: pd.DataFrame) -> Dict:
        """
        Genera estadísticas descriptivas del DataFrame.

        Args:
            df: DataFrame maestro

        Returns:
            Diccionario con estadísticas clave
        """
        return {
            "total_registros": len(df),
            "fecha_inicio": df.index.min().strftime("%Y-%m-%d"),
            "fecha_fin": df.index.max().strftime("%Y-%m-%d"),
            "años_unicos": int(df['año'].nunique()),
            "valores_faltantes": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
            "estadisticas_temperatura": {
                "T2M_MAX_promedio": round(df['T2M_MAX'].mean(), 2),
                "T2M_MIN_promedio": round(df['T2M_MIN'].mean(), 2),
                "T2M_MAX_max": round(df['T2M_MAX'].max(), 2),
                "T2M_MIN_min": round(df['T2M_MIN'].min(), 2)
            }
        }
