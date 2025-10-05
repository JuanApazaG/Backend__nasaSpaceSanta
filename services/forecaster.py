"""
Implementación de pronóstico usando modelos ARIMA
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


class ARIMAForecaster:
    """
    Implementación de pronóstico usando modelos ARIMA.
    Requisito obligatorio del desafío NASA Space Apps.
    """

    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0)):
        """
        Inicializa el forecaster con parámetros ARIMA.

        Args:
            order: Tupla (p, d, q) para el modelo ARIMA
        """
        self.order = order
        self.model = None
        self.model_fit = None

    def preparar_serie_temporal(
        self,
        df: pd.DataFrame,
        metrica: str
    ) -> pd.Series:
        """
        Prepara una serie temporal para el modelo ARIMA.

        Args:
            df: DataFrame con datos
            metrica: Columna a usar como serie temporal

        Returns:
            Serie temporal preparada
        """
        serie = df[metrica].dropna()
        return serie

    def entrenar_modelo(
        self,
        df: pd.DataFrame,
        metrica: str
    ) -> Dict:
        """
        Entrena el modelo ARIMA con datos históricos.

        Args:
            df: DataFrame con datos históricos
            metrica: Métrica a pronosticar

        Returns:
            Diccionario con información del entrenamiento
        """
        print(f"🔄 Entrenando modelo ARIMA{self.order} para {metrica}...")

        # Preparar datos
        serie = self.preparar_serie_temporal(df, metrica)

        # Entrenar modelo
        try:
            self.model = ARIMA(serie, order=self.order)
            self.model_fit = self.model.fit()

            # Resumen del modelo
            aic = self.model_fit.aic
            bic = self.model_fit.bic

            print(f"✅ Modelo entrenado exitosamente")
            print(f"   AIC: {aic:.2f}, BIC: {bic:.2f}")

            return {
                "exito": True,
                "aic": round(aic, 2),
                "bic": round(bic, 2),
                "parametros": self.order,
                "observaciones_entrenamiento": len(serie)
            }

        except Exception as e:
            print(f"❌ Error entrenando modelo: {e}")
            return {"exito": False, "error": str(e)}

    def pronosticar(
        self,
        pasos: int = 7
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera pronósticos usando el modelo entrenado.

        Args:
            pasos: Número de pasos futuros a pronosticar

        Returns:
            Tupla (pronóstico, intervalo_inferior, intervalo_superior)
        """
        if self.model_fit is None:
            raise ValueError("Modelo no ha sido entrenado. Llama a entrenar_modelo() primero.")

        # Generar pronóstico
        forecast = self.model_fit.forecast(steps=pasos)

        # Obtener intervalos de confianza (95%)
        forecast_result = self.model_fit.get_forecast(steps=pasos)
        conf_int = forecast_result.conf_int(alpha=0.05)

        return forecast, conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values

    @staticmethod
    def calculate_rmse(actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        """
        Calcula el Root Mean Squared Error (RMSE) entre valores reales y predichos.

        Args:
            actual_values: Array o Serie de valores reales.
            predicted_values: Array o Serie de valores predichos.

        Returns:
            El valor del RMSE.
        """
        # Asegurar que son arrays de numpy
        actual_values = np.asarray(actual_values)
        predicted_values = np.asarray(predicted_values)

        # Calcular MSE
        mse = mean_squared_error(actual_values, predicted_values)

        # Calcular RMSE
        rmse = np.sqrt(mse)

        return rmse
