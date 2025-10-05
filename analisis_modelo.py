"""
=============================================================================
ANÁLISIS DE EXACTITUD DEL MODELO ARIMA
=============================================================================
Este script evalúa la precisión y exactitud del modelo ARIMA utilizado
en el sistema de pronóstico climático.

Métricas calculadas:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score
- Análisis de residuos
- Comparación con modelo naive (baseline)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Importar servicios
from services.translator import UserInputTranslator
from services.nasa_api import NASAPowerAPI
from services.data_processor import DataProcessor
from services.forecaster import ARIMAForecaster
from sklearn.metrics import mean_absolute_error, r2_score

class ModeloAnalyzer:
    """Analizador de exactitud del modelo ARIMA"""
    
    def __init__(self):
        self.traductor = UserInputTranslator()
        self.api_cliente = NASAPowerAPI()
        self.procesador = DataProcessor()
        self.forecaster = ARIMAForecaster(order=(5, 1, 0))
        
    def calcular_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula Mean Absolute Percentage Error (MAPE).
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            
        Returns:
            MAPE en porcentaje
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Evitar división por cero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def modelo_naive_baseline(self, serie: pd.Series, pasos: int) -> np.ndarray:
        """
        Implementa un modelo naive (último valor observado se repite).
        Usado como baseline de comparación.
        
        Args:
            serie: Serie temporal de entrenamiento
            pasos: Número de pasos a predecir
            
        Returns:
            Array con predicciones naive
        """
        ultimo_valor = serie.iloc[-1]
        return np.array([ultimo_valor] * pasos)
    
    def analizar_residuos(self, residuos: np.ndarray) -> Dict:
        """
        Analiza los residuos del modelo para detectar problemas.
        
        Args:
            residuos: Errores del modelo (y_true - y_pred)
            
        Returns:
            Diccionario con estadísticas de residuos
        """
        return {
            "media": round(np.mean(residuos), 4),
            "mediana": round(np.median(residuos), 4),
            "desviacion_estandar": round(np.std(residuos), 4),
            "minimo": round(np.min(residuos), 4),
            "maximo": round(np.max(residuos), 4),
            "asimetria": round(float(pd.Series(residuos).skew()), 4),
            "curtosis": round(float(pd.Series(residuos).kurtosis()), 4)
        }
    
    def evaluar_modelo_completo(
        self,
        ciudad: str = "Santa Cruz de la Sierra, Bolivia",
        evento: str = "Halloween",
        metrica: str = "PRECTOTCORR",
        año_objetivo: int = 2025
    ) -> Dict:
        """
        Evalúa completamente el modelo ARIMA con datos reales.
        
        Args:
            ciudad: Ciudad para análisis
            evento: Evento para análisis
            metrica: Métrica climática (PRECTOTCORR, T2M_MAX, etc.)
            año_objetivo: Año objetivo
            
        Returns:
            Diccionario con todas las métricas de evaluación
        """
        print("="*80)
        print("🔍 ANÁLISIS DE EXACTITUD DEL MODELO ARIMA")
        print("="*80)
        
        # 1. Obtener datos
        print("\n📡 Obteniendo datos de NASA POWER API...")
        ubicacion = self.traductor.geocodificar_ciudad(ciudad)
        fechas = self.traductor.interpretar_fecha(evento, año_objetivo)
        
        respuestas_api = self.api_cliente.consultar_historico_multianio(
            latitud=ubicacion['latitud'],
            longitud=ubicacion['longitud'],
            fecha_inicio=fechas['fecha_inicio'],
            fecha_fin=fechas['fecha_fin'],
            años_historicos=list(range(2005, año_objetivo))
        )
        
        df_historico = self.procesador.crear_dataframe_maestro(respuestas_api)
        print(f"✅ Datos obtenidos: {len(df_historico)} registros")
        
        # 2. Dividir en train/test (últimos 7 días como test)
        test_size = 7
        df_train = df_historico.iloc[:-test_size]
        df_test = df_historico.iloc[-test_size:]
        
        print(f"\n📊 División de datos:")
        print(f"   Entrenamiento: {len(df_train)} días")
        print(f"   Prueba: {len(df_test)} días")
        
        # 3. Entrenar modelo ARIMA
        print("\n🤖 Entrenando modelo ARIMA(5,1,0)...")
        entrenamiento = self.forecaster.entrenar_modelo(df=df_train, metrica=metrica)
        
        if not entrenamiento["exito"]:
            return {"error": "No se pudo entrenar el modelo", "detalle": entrenamiento}
        
        # 4. Generar predicciones sobre conjunto de test
        print("\n🔮 Generando predicciones...")
        predictions_arima = self.forecaster.model_fit.predict(
            start=len(df_train),
            end=len(df_train) + test_size - 1
        )
        
        y_true = df_test[metrica].values
        y_pred_arima = predictions_arima.values
        
        # 5. Modelo baseline (naive)
        serie_train = df_train[metrica].dropna()
        y_pred_naive = self.modelo_naive_baseline(serie_train, test_size)
        
        # 6. Calcular métricas para ARIMA
        print("\n📈 Calculando métricas de exactitud...")
        rmse_arima = self.forecaster.calculate_rmse(y_true, y_pred_arima)
        mae_arima = mean_absolute_error(y_true, y_pred_arima)
        mape_arima = self.calcular_mape(y_true, y_pred_arima)
        r2_arima = r2_score(y_true, y_pred_arima)
        
        # 7. Calcular métricas para modelo naive (baseline)
        rmse_naive = self.forecaster.calculate_rmse(y_true, y_pred_naive)
        mae_naive = mean_absolute_error(y_true, y_pred_naive)
        mape_naive = self.calcular_mape(y_true, y_pred_naive)
        r2_naive = r2_score(y_true, y_pred_naive)
        
        # 8. Análisis de residuos
        residuos_arima = y_true - y_pred_arima
        analisis_residuos = self.analizar_residuos(residuos_arima)
        
        # 9. Calcular mejora sobre baseline
        mejora_rmse = ((rmse_naive - rmse_arima) / rmse_naive) * 100
        mejora_mae = ((mae_naive - mae_arima) / mae_naive) * 100
        
        # 10. Compilar resultados
        resultados = {
            "configuracion": {
                "ciudad": ciudad,
                "evento": evento,
                "metrica": metrica,
                "modelo": "ARIMA(5,1,0)",
                "tamaño_entrenamiento": len(df_train),
                "tamaño_prueba": test_size
            },
            "metricas_arima": {
                "RMSE": round(rmse_arima, 4),
                "MAE": round(mae_arima, 4),
                "MAPE": round(mape_arima, 2),
                "R2_Score": round(r2_arima, 4),
                "AIC": entrenamiento["aic"],
                "BIC": entrenamiento["bic"]
            },
            "metricas_baseline_naive": {
                "RMSE": round(rmse_naive, 4),
                "MAE": round(mae_naive, 4),
                "MAPE": round(mape_naive, 2),
                "R2_Score": round(r2_naive, 4)
            },
            "mejora_sobre_baseline": {
                "mejora_RMSE_porcentaje": round(mejora_rmse, 2),
                "mejora_MAE_porcentaje": round(mejora_mae, 2)
            },
            "analisis_residuos": analisis_residuos,
            "valores_reales_vs_predichos": {
                "y_true": y_true.tolist(),
                "y_pred_arima": y_pred_arima.tolist(),
                "y_pred_naive": y_pred_naive.tolist(),
                "fechas_test": [d.strftime('%Y-%m-%d') for d in df_test.index]
            },
            "interpretacion": self.interpretar_metricas(rmse_arima, mae_arima, mape_arima, r2_arima, metrica)
        }
        
        # 11. Imprimir resumen
        self.imprimir_resumen(resultados)
        
        # 12. Generar visualizaciones
        self.visualizar_resultados(resultados)
        
        return resultados
    
    def interpretar_metricas(self, rmse: float, mae: float, mape: float, r2: float, metrica: str) -> Dict:
        """
        Interpreta las métricas y proporciona conclusiones sobre la exactitud.
        
        Args:
            rmse: Root Mean Squared Error
            mae: Mean Absolute Error
            mape: Mean Absolute Percentage Error
            r2: R² Score
            metrica: Nombre de la métrica evaluada
            
        Returns:
            Diccionario con interpretaciones
        """
        interpretacion = {
            "calidad_general": "",
            "precision_promedio": f"El modelo se equivoca en promedio {mae:.2f} unidades",
            "variabilidad_explicada": f"El modelo explica {r2*100:.1f}% de la variabilidad",
            "recomendacion": ""
        }
        
        # Interpretar R²
        if r2 >= 0.9:
            interpretacion["calidad_general"] = "EXCELENTE: El modelo tiene muy alta precisión"
        elif r2 >= 0.7:
            interpretacion["calidad_general"] = "BUENA: El modelo tiene precisión aceptable"
        elif r2 >= 0.5:
            interpretacion["calidad_general"] = "MODERADA: El modelo tiene precisión limitada"
        else:
            interpretacion["calidad_general"] = "BAJA: El modelo necesita mejoras significativas"
        
        # Interpretar MAPE
        if mape < 10:
            interpretacion["recomendacion"] = "El modelo es muy preciso y confiable para uso en producción"
        elif mape < 20:
            interpretacion["recomendacion"] = "El modelo es aceptable para pronósticos generales"
        elif mape < 50:
            interpretacion["recomendacion"] = "El modelo tiene errores significativos, usar con precaución"
        else:
            interpretacion["recomendacion"] = "El modelo no es confiable, considerar métodos alternativos"
        
        return interpretacion
    
    def imprimir_resumen(self, resultados: Dict):
        """Imprime un resumen legible de los resultados."""
        print("\n" + "="*80)
        print("📊 RESUMEN DE RESULTADOS")
        print("="*80)
        
        print(f"\n🎯 Configuración:")
        print(f"   Modelo: {resultados['configuracion']['modelo']}")
        print(f"   Métrica evaluada: {resultados['configuracion']['metrica']}")
        print(f"   Datos de entrenamiento: {resultados['configuracion']['tamaño_entrenamiento']} días")
        print(f"   Datos de prueba: {resultados['configuracion']['tamaño_prueba']} días")
        
        print(f"\n🤖 Métricas del Modelo ARIMA:")
        metricas = resultados['metricas_arima']
        print(f"   RMSE (Error cuadrático medio): {metricas['RMSE']}")
        print(f"   MAE (Error absoluto medio): {metricas['MAE']}")
        print(f"   MAPE (Error porcentual): {metricas['MAPE']}%")
        print(f"   R² Score (Varianza explicada): {metricas['R2_Score']}")
        print(f"   AIC (Criterio de información): {metricas['AIC']}")
        print(f"   BIC (Criterio bayesiano): {metricas['BIC']}")
        
        print(f"\n📊 Comparación con Modelo Naive (Baseline):")
        mejora = resultados['mejora_sobre_baseline']
        print(f"   Mejora en RMSE: {mejora['mejora_RMSE_porcentaje']:.2f}%")
        print(f"   Mejora en MAE: {mejora['mejora_MAE_porcentaje']:.2f}%")
        
        if mejora['mejora_RMSE_porcentaje'] > 0:
            print(f"   ✅ ARIMA es {mejora['mejora_RMSE_porcentaje']:.1f}% mejor que el baseline")
        else:
            print(f"   ⚠️ ARIMA es {abs(mejora['mejora_RMSE_porcentaje']):.1f}% peor que el baseline")
        
        print(f"\n🔬 Análisis de Residuos:")
        residuos = resultados['analisis_residuos']
        print(f"   Media de errores: {residuos['media']} (cercano a 0 es ideal)")
        print(f"   Desviación estándar: {residuos['desviacion_estandar']}")
        print(f"   Rango de errores: [{residuos['minimo']}, {residuos['maximo']}]")
        
        print(f"\n💡 Interpretación:")
        interp = resultados['interpretacion']
        print(f"   {interp['calidad_general']}")
        print(f"   {interp['precision_promedio']}")
        print(f"   {interp['variabilidad_explicada']}")
        print(f"   Recomendación: {interp['recomendacion']}")
        
        print("\n" + "="*80)
    
    def visualizar_resultados(self, resultados: Dict):
        """Genera visualizaciones de los resultados."""
        valores = resultados['valores_reales_vs_predichos']
        fechas = valores['fechas_test']
        y_true = valores['y_true']
        y_pred_arima = valores['y_pred_arima']
        y_pred_naive = valores['y_pred_naive']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análisis de Exactitud del Modelo ARIMA', fontsize=16, fontweight='bold')
        
        # 1. Predicciones vs Valores Reales
        ax1 = axes[0, 0]
        ax1.plot(fechas, y_true, 'o-', label='Valores Reales', color='green', linewidth=2, markersize=8)
        ax1.plot(fechas, y_pred_arima, 's--', label='ARIMA', color='blue', linewidth=2, markersize=6)
        ax1.plot(fechas, y_pred_naive, '^--', label='Naive Baseline', color='red', linewidth=1, markersize=6)
        ax1.set_title('Predicciones vs Valores Reales')
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel(resultados['configuracion']['metrica'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Residuos (errores)
        ax2 = axes[0, 1]
        residuos = np.array(y_true) - np.array(y_pred_arima)
        ax2.scatter(range(len(residuos)), residuos, color='purple', alpha=0.6, s=100)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('Residuos del Modelo ARIMA')
        ax2.set_xlabel('Observación')
        ax2.set_ylabel('Error (Real - Predicho)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Histograma de residuos
        ax3 = axes[1, 0]
        ax3.hist(residuos, bins=10, color='orange', alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Distribución de Residuos')
        ax3.set_xlabel('Error')
        ax3.set_ylabel('Frecuencia')
        ax3.grid(True, alpha=0.3)
        
        # 4. Comparación de métricas
        ax4 = axes[1, 1]
        metricas_nombres = ['RMSE', 'MAE', 'MAPE']
        arima_vals = [
            resultados['metricas_arima']['RMSE'],
            resultados['metricas_arima']['MAE'],
            resultados['metricas_arima']['MAPE']
        ]
        naive_vals = [
            resultados['metricas_baseline_naive']['RMSE'],
            resultados['metricas_baseline_naive']['MAE'],
            resultados['metricas_baseline_naive']['MAPE']
        ]
        
        x = np.arange(len(metricas_nombres))
        width = 0.35
        ax4.bar(x - width/2, arima_vals, width, label='ARIMA', color='blue', alpha=0.7)
        ax4.bar(x + width/2, naive_vals, width, label='Naive', color='red', alpha=0.7)
        ax4.set_title('Comparación ARIMA vs Baseline')
        ax4.set_xlabel('Métrica')
        ax4.set_ylabel('Valor')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metricas_nombres)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('analisis_modelo_arima.png', dpi=300, bbox_inches='tight')
        print("\n💾 Gráficos guardados en: analisis_modelo_arima.png")
        plt.show()


if __name__ == "__main__":
    # Crear analizador
    analyzer = ModeloAnalyzer()
    
    # Evaluar modelo con datos reales
    resultados = analyzer.evaluar_modelo_completo(
        ciudad="Santa Cruz de la Sierra, Bolivia",
        evento="Halloween",
        metrica="PRECTOTCORR",  # Precipitación
        año_objetivo=2025
    )
    
    # También puedes probar con otras métricas:
    # metrica="T2M_MAX"  # Temperatura máxima
    # metrica="WS10M_MAX"  # Velocidad del viento
    
    print("\n✅ Análisis completado!")
