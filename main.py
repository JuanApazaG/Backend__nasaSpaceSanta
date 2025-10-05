"""
=============================================================================
NASA SPACE APPS CHALLENGE 2025
"¿Qué significa la 'P' en 'Clima'? ¡Probabilidad!"
=============================================================================

FastAPI Application - Sistema de Análisis Climático Probabilístico
Autor: Arquitecto de Software Senior
Fecha: Octubre 2025
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List
import warnings

from models.schemas import AnalisisRequest, EventoInfo, CondicionInfo
from models.responses import AnalisisResponse, ErrorResponse
from services.translator import UserInputTranslator
from services.nasa_api import NASAPowerAPI
from services.data_processor import DataProcessor
from services.probability import ProbabilityAnalyzer
from services.forecaster import ARIMAForecaster

warnings.filterwarnings('ignore')

# Inicializar FastAPI
app = FastAPI(
    title="NASA Climate Probability API",
    description="Sistema de Análisis Climático Probabilístico con NASA POWER API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar servicios globales
traductor = UserInputTranslator()
api_cliente = NASAPowerAPI()
procesador = DataProcessor()
analizador_prob = ProbabilityAnalyzer()
forecaster = ARIMAForecaster(order=(5, 1, 0))


@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz - Información de la API"""
    return {
        "nombre": "NASA Climate Probability API",
        "version": "1.0.0",
        "descripcion": "Sistema de Análisis Climático Probabilístico",
        "documentacion": "/docs",
        "endpoints": {
            "analisis": "/api/v1/analisis",
            "eventos": "/api/v1/eventos",
            "condiciones": "/api/v1/condiciones"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Endpoint de health check"""
    return {
        "status": "healthy",
        "servicios": {
            "traductor": "ok",
            "nasa_api": "ok",
            "procesador": "ok",
            "analizador": "ok",
            "forecaster": "ok"
        }
    }


@app.get("/api/v1/eventos", response_model=List[EventoInfo], tags=["Catálogos"])
async def listar_eventos():
    """
    Lista todos los eventos disponibles para consulta
    """
    eventos = []
    for nombre, data in traductor.eventos.items():
        eventos.append(EventoInfo(
            nombre=nombre,
            mes=data["mes"],
            dia=data["dia"]
        ))
    return eventos


@app.get("/api/v1/condiciones", response_model=List[CondicionInfo], tags=["Catálogos"])
async def listar_condiciones():
    """
    Lista todas las condiciones climáticas disponibles
    """
    condiciones = []
    for nombre, data in traductor.condiciones.items():
        condiciones.append(CondicionInfo(
            nombre=nombre,
            metrica=data["metrica"],
            umbral=data["umbral"],
            condicion=data["condicion"],
            descripcion=data["descripcion"]
        ))
    return condiciones


@app.post("/api/v1/analisis", response_model=AnalisisResponse, tags=["Análisis"])
async def ejecutar_analisis(request: AnalisisRequest):
    """
    Ejecuta un análisis climático completo
    
    Este endpoint implementa el pipeline completo:
    1. Traduce la entrada del usuario
    2. Consulta la API de NASA POWER
    3. Procesa los datos históricos
    4. Calcula probabilidades
    5. Genera pronóstico ARIMA con evaluación RMSE
    
    **Parámetros:**
    - ciudad: Nombre de la ciudad (ej. "Santa Cruz de la Sierra, Bolivia")
    - evento: Nombre del evento o fecha (ej. "Halloween")
    - condicion_texto: Condición climática (ej. "llueva", "caluroso")
    - año_objetivo: Año para el análisis (default: 2025)
    - visualizar: No aplica en API (siempre False)
    
    **Retorna:**
    - Análisis completo con probabilidades, pronóstico y métricas
    """
    
    try:
        # 1. Traducir entrada del usuario
        ubicacion = traductor.geocodificar_ciudad(request.ciudad)
        if not ubicacion:
            raise HTTPException(
                status_code=400,
                detail=f"No se pudo geocodificar la ciudad: {request.ciudad}"
            )
        
        fechas = traductor.interpretar_fecha(request.evento, request.año_objetivo)
        if not fechas:
            raise HTTPException(
                status_code=400,
                detail=f"No se pudo interpretar el evento/fecha: {request.evento}"
            )
        
        condicion_params = traductor.mapear_condicion(request.condicion_texto)
        if not condicion_params:
            raise HTTPException(
                status_code=400,
                detail=f"No se pudo mapear la condición: {request.condicion_texto}"
            )
        
        # 2. Consultar NASA POWER API
        respuestas_api = api_cliente.consultar_historico_multianio(
            latitud=ubicacion['latitud'],
            longitud=ubicacion['longitud'],
            fecha_inicio=fechas['fecha_inicio'],
            fecha_fin=fechas['fecha_fin'],
            años_historicos=list(range(2005, request.año_objetivo))
        )
        
        if not respuestas_api:
            raise HTTPException(
                status_code=503,
                detail="No se pudieron obtener datos de la API de NASA POWER"
            )
        
        # 3. Procesar datos
        df_historico = procesador.crear_dataframe_maestro(respuestas_api)
        if df_historico.empty:
            raise HTTPException(
                status_code=500,
                detail="El DataFrame histórico está vacío después del procesamiento"
            )
        
        estadisticas_df = procesador.generar_estadisticas(df_historico)
        
        # 4. Calcular probabilidad histórica
        probabilidad_historica = analizador_prob.calcular_probabilidad(
            df=df_historico,
            metrica=condicion_params['metrica'],
            umbral=condicion_params['umbral'],
            condicion=condicion_params['condicion']
        )
        
        probabilidad_mensual_df = analizador_prob.analisis_por_mes(
            df=df_historico,
            metrica=condicion_params['metrica'],
            umbral=condicion_params['umbral'],
            condicion=condicion_params['condicion']
        )
        
        # 5. Generar pronóstico ARIMA con evaluación
        from datetime import datetime, timedelta
        
        fecha_central_obj = datetime.strptime(fechas['fecha_central'], "%Y-%m-%d")
        fecha_inicio_pron = fecha_central_obj - timedelta(days=3)
        fecha_fin_pron = fecha_central_obj + timedelta(days=3)
        dias_pronostico = (fecha_fin_pron - fecha_inicio_pron).days + 1
        
        # Dividir datos en entrenamiento y prueba
        test_size = dias_pronostico
        if len(df_historico) < test_size:
            raise HTTPException(
                status_code=400,
                detail=f"Datos históricos insuficientes ({len(df_historico)} días) para conjunto de prueba de {test_size} días"
            )
        
        df_train = df_historico.iloc[:-test_size]
        df_test = df_historico.iloc[-test_size:]
        
        # Entrenar modelo
        entrenamiento_arima = forecaster.entrenar_modelo(
            df=df_train,
            metrica=condicion_params['metrica']
        )
        
        if not entrenamiento_arima["exito"]:
            raise HTTPException(
                status_code=500,
                detail=f"No se pudo entrenar el modelo ARIMA: {entrenamiento_arima.get('error', 'Error desconocido')}"
            )
        
        # Evaluar en conjunto de prueba
        predictions_test = forecaster.model_fit.predict(
            start=len(df_train),
            end=len(df_train) + test_size - 1
        )
        actual_values_test = df_test[condicion_params['metrica']].values
        rmse_test = forecaster.calculate_rmse(actual_values_test, predictions_test)
        
        # Generar pronóstico
        import pandas as pd
        pronostico_valores, pronostico_inf, pronostico_sup = forecaster.pronosticar(pasos=dias_pronostico)
        
        fechas_pronostico = pd.date_range(
            start=fecha_inicio_pron,
            periods=dias_pronostico,
            freq='D'
        )
        
        df_pronostico = pd.DataFrame({
            'fecha': fechas_pronostico,
            'pronostico': pronostico_valores,
            'intervalo_inferior': pronostico_inf,
            'intervalo_superior': pronostico_sup
        }).set_index('fecha')
        
        # Calcular si la condición se cumple
        pronostico_cumple_condicion = False
        if condicion_params['condicion'] == "mayor_que":
            pronostico_cumple_condicion = (df_pronostico['pronostico'] > condicion_params['umbral']).any()
        elif condicion_params['condicion'] == "menor_que":
            pronostico_cumple_condicion = (df_pronostico['pronostico'] < condicion_params['umbral']).any()
        elif condicion_params['condicion'] == "igual_a":
            pronostico_cumple_condicion = (df_pronostico['pronostico'] == condicion_params['umbral']).any()
        
        # Convertir Timestamps a strings para serialización JSON
        df_pronostico_serializable = df_pronostico.reset_index()
        df_pronostico_serializable['fecha'] = df_pronostico_serializable['fecha'].dt.strftime('%Y-%m-%d')
        
        # Convertir tipos numpy a tipos nativos de Python
        datos_pronosticados = []
        for _, row in df_pronostico_serializable.iterrows():
            datos_pronosticados.append({
                'fecha': row['fecha'],
                'pronostico': float(row['pronostico']),
                'intervalo_inferior': float(row['intervalo_inferior']),
                'intervalo_superior': float(row['intervalo_superior'])
            })
        
        # 6. Compilar resultados
        resultado_final = {
            "estado": "exito",
            "parametros_entrada": {
                "ciudad": request.ciudad,
                "evento": request.evento,
                "condicion_texto": request.condicion_texto,
                "año_objetivo": request.año_objetivo
            },
            "ubicacion_geocodificada": ubicacion,
            "rango_fechas_evento": fechas,
            "condicion_mapeada": condicion_params,
            "estadisticas_historicas_generales": estadisticas_df,
            "probabilidad_historica_evento": probabilidad_historica,
            "probabilidad_historica_mensual": probabilidad_mensual_df.to_dict(orient='records'),
            "pronostico_arima": {
                "parametros_modelo": entrenamiento_arima,
                "rmse_evaluacion": round(rmse_test, 2),
                "datos_pronosticados": datos_pronosticados,
                "condicion_cumplida_en_pronostico": bool(pronostico_cumple_condicion)
            },
            "nota": "La probabilidad histórica se basa en la frecuencia observada en años anteriores. El pronóstico ARIMA es una estimación para el año objetivo."
        }
        
        return JSONResponse(content=resultado_final)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
