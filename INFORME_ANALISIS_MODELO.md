# 📊 INFORME DE ANÁLISIS DE EXACTITUD DEL MODELO ARIMA

## Resumen Ejecutivo

El modelo ARIMA(5,1,0) implementado para pronóstico de precipitación **presenta problemas significativos de exactitud** y **no es confiable** en su configuración actual.

---

## 🔴 Problemas Identificados

### 1. **Métricas de Exactitud Muy Bajas**

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **RMSE** | 3.26 | Error cuadrático medio alto |
| **MAE** | 2.43 | Se equivoca en promedio 2.43 mm de precipitación |
| **MAPE** | 76.06% | Error porcentual extremadamente alto |
| **R² Score** | -0.93 | **CRÍTICO**: Valor negativo indica que el modelo es peor que una línea horizontal (media) |

### 2. **Peor que el Modelo Baseline (Naive)**

- El modelo ARIMA es **8.2% peor** en RMSE que simplemente repetir el último valor observado
- El modelo ARIMA es **13% peor** en MAE que el baseline
- **Conclusión**: El modelo ARIMA no está añadiendo valor predictivo

### 3. **Residuos con Sesgo**

| Estadística | Valor | Problema |
|-------------|-------|----------|
| Media de errores | 2.27 | **Sesgo positivo** (debería ser ~0) - El modelo subestima sistemáticamente |
| Desviación estándar | 2.34 | Alta variabilidad en los errores |
| Rango | [-0.58, 5.99] | Algunos errores son 6 veces el valor medio |

---

## 🔍 Causas Raíz del Problema

### 1. **Datos Insuficientes para el Modelo**
- Solo 140 observaciones (20 años × 7 días alrededor de Halloween)
- ARIMA necesita series temporales largas y continuas
- Los datos están fragmentados (7 días por año, no continuos)

### 2. **Estacionalidad No Capturada**
- ARIMA(5,1,0) no tiene componente estacional (no hay término S)
- La precipitación tiene patrones estacionales fuertes
- Debería usarse SARIMA (Seasonal ARIMA)

### 3. **Parámetros No Optimizados**
- Los parámetros (5,1,0) parecen fijos arbitrariamente
- No hay evidencia de optimización (prueba de múltiples combinaciones)
- AIC=916 y BIC=933 son valores altos (indicando mal ajuste)

### 4. **Naturaleza de la Variable**
- La precipitación es altamente estocástica (aleatoria)
- Tiene muchos valores cero (días sin lluvia)
- Es difícil de predecir incluso con modelos sofisticados

---

## 📈 Comparación de Métricas

```
Modelo          | RMSE  | MAE   | MAPE    | R²     | Recomendación
----------------|-------|-------|---------|--------|---------------
ARIMA(5,1,0)    | 3.26  | 2.43  | 76.06%  | -0.93  | ❌ NO USAR
Naive Baseline  | 3.01  | 2.15  | ~65%    | -0.65  | ⚠️ Mejor que ARIMA
Ideal esperado  | <1.0  | <0.8  | <20%    | >0.7   | ✅ Objetivo
```

---

## ✅ Recomendaciones para Mejorar el Modelo

### Soluciones a Corto Plazo (Rápidas)

#### 1. **Usar el Modelo Naive como Baseline**
```python
# Es más preciso que ARIMA actualmente
prediction = last_observed_value
```

#### 2. **Ajustar Parámetros ARIMA Automáticamente**
```python
from pmdarima import auto_arima

# Encuentra los mejores parámetros automáticamente
modelo = auto_arima(
    serie_temporal,
    start_p=0, max_p=5,
    start_q=0, max_q=5,
    d=None,  # Determina d automáticamente
    seasonal=True,  # Incluir estacionalidad
    m=365,  # Período estacional (días por año)
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)
```

#### 3. **Probar SARIMA (ARIMA con Estacionalidad)**
```python
# Incluir componente estacional
forecaster = ARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
```

### Soluciones a Medio Plazo (Mejoras Significativas)

#### 4. **Ensamble de Modelos**
Combinar múltiples modelos:
```python
# Promedio ponderado de varios modelos
prediction = (
    0.3 * arima_pred +
    0.3 * sarima_pred +
    0.2 * prophet_pred +
    0.2 * xgboost_pred
)
```

#### 5. **Usar Prophet (de Facebook)**
```python
from prophet import Prophet

# Mejor para series con estacionalidad y tendencias
modelo = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)
modelo.fit(df)
```

#### 6. **Modelos de Machine Learning**
```python
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Usar features derivados (mes, día, año, promedios históricos)
features = ['mes', 'dia', 'dia_año', 'promedio_historico', 'tendencia']
modelo = XGBRegressor()
modelo.fit(X_train[features], y_train)
```

### Soluciones a Largo Plazo (Óptimas)

#### 7. **Redes Neuronales (LSTM/GRU)**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Para series temporales complejas
modelo = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
    Dense(1)
])
```

#### 8. **Modelos Probabilísticos Bayesianos**
```python
import pymc3 as pm

# Captura incertidumbre mejor
with pm.Model() as modelo:
    # Definir prior y likelihood
    trace = pm.sample(1000)
```

---

## 🎯 Estrategia Recomendada (Implementación Inmediata)

### Paso 1: Usar Probabilidad Histórica en Lugar de ARIMA

Para el caso de uso actual ("¿lloverá en mi evento?"), **la probabilidad histórica es más confiable**:

```python
# Esto ya lo tienes implementado y es MEJOR que ARIMA
probabilidad_historica = (dias_con_lluvia / dias_totales) * 100

# Ejemplo: "Históricamente llueve en Halloween el 35% de las veces"
```

**Ventajas**:
- ✅ Más interpretable
- ✅ Más preciso para este caso de uso
- ✅ No asume patrones que no existen

### Paso 2: Añadir Intervalos de Confianza

```python
from scipy import stats

# Calcular intervalo de confianza para la probabilidad
n = dias_totales
p = probabilidad
intervalo = stats.binom.interval(0.95, n, p)

# "Hay 35% de probabilidad de lluvia (95% CI: 28%-42%)"
```

### Paso 3: Mejorar ARIMA Solo Si Es Necesario

Si realmente necesitas pronósticos punto-a-punto:

```python
# 1. Obtener más datos (series continuas, no fragmentadas)
años_completos = range(2010, 2025)  # 15 años completos

# 2. Usar auto_arima para optimizar parámetros
from pmdarima import auto_arima
modelo_optimizado = auto_arima(serie, seasonal=True, m=365)

# 3. Validar con cross-validation
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
rmses = []
for train_idx, test_idx in tscv.split(serie):
    # Entrenar y evaluar
    rmses.append(calcular_rmse(...))

print(f"RMSE promedio: {np.mean(rmses)}")
```

---

## 📊 Métricas Objetivo

Para que el modelo sea "bueno" y confiable:

| Métrica | Objetivo | Actual | Gap |
|---------|----------|--------|-----|
| RMSE | < 1.0 mm | 3.26 mm | ❌ 226% peor |
| MAE | < 0.8 mm | 2.43 mm | ❌ 204% peor |
| MAPE | < 20% | 76% | ❌ 280% peor |
| R² | > 0.7 | -0.93 | ❌ -233% peor |

---

## 🔧 Código de Mejora Inmediata

### Opción A: Usar solo probabilidad histórica (RECOMENDADO)

```python
@app.post("/api/v1/analisis")
async def ejecutar_analisis(request: AnalisisRequest):
    # ... código existente ...
    
    # EN LUGAR DE USAR ARIMA, usar solo probabilidad histórica
    resultado_final = {
        "probabilidad_historica": probabilidad_historica,
        "interpretacion": f"Basado en {años_unicos} años de datos, "
                         f"la condición se cumple el {probabilidad}% de las veces",
        "confianza": "ALTA" if dias_totales > 50 else "MEDIA",
        # REMOVER o marcar como experimental:
        "pronostico_arima": {
            "ADVERTENCIA": "Modelo experimental con baja precisión",
            "recomendacion": "Usar probabilidad histórica en su lugar"
        }
    }
```

### Opción B: Mejorar ARIMA con auto_arima

```python
# En services/forecaster.py
from pmdarima import auto_arima

class ARIMAForecaster:
    def entrenar_modelo(self, df, metrica):
        serie = self.preparar_serie_temporal(df, metrica)
        
        # Auto-optimizar parámetros
        self.model_fit = auto_arima(
            serie,
            start_p=0, max_p=5,
            start_q=0, max_q=5,
            seasonal=True,
            m=7,  # Periodicidad semanal
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        
        return {
            "exito": True,
            "parametros": self.model_fit.order,
            "aic": self.model_fit.aic(),
            "bic": self.model_fit.bic()
        }
```

---

## 📝 Conclusión Final

### Estado Actual del Modelo ARIMA
- ❌ **No es confiable** para uso en producción
- ❌ **Peor que métodos simples** (naive baseline)
- ❌ **R² negativo** indica que no captura patrones
- ❌ **Errores sistemáticos** (sesgo de 2.27 mm)

### Recomendación Principal
**Usar la probabilidad histórica** (que ya está implementada) como respuesta principal, y:
- Marcar ARIMA como "experimental" o removerlo temporalmente
- Implementar mejoras (auto_arima, Prophet, ML) antes de usarlo en producción
- Validar cualquier modelo nuevo con métricas > 0.7 R² antes de deployment

### Acción Inmediata
```python
# En la respuesta de la API, priorizar:
return {
    "respuesta_principal": probabilidad_historica,  # CONFIABLE ✅
    "confianza": "ALTA",
    "pronostico_arima": None,  # Deshabilitado hasta mejora
    "nota": "Basado en análisis de frecuencia histórica de 20 años"
}
```

---

**Última actualización**: Octubre 2025  
**Analista**: Sistema de Evaluación Automática  
**Estado**: ❌ Modelo no apto para producción - Requiere mejoras significativas
