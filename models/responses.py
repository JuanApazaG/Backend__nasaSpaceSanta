"""
Modelos de respuesta para la API
"""

from pydantic import BaseModel
from typing import Dict, List, Any, Optional


class ErrorResponse(BaseModel):
    """Respuesta de error estándar"""
    estado: str = "error"
    mensaje: str
    detalle: Optional[Dict[str, Any]] = None


class AnalisisResponse(BaseModel):
    """Respuesta del análisis climático completo"""
    estado: str
    parametros_entrada: Dict[str, Any]
    ubicacion_geocodificada: Dict[str, Any]
    rango_fechas_evento: Dict[str, str]
    condicion_mapeada: Dict[str, Any]
    estadisticas_historicas_generales: Dict[str, Any]
    probabilidad_historica_evento: Dict[str, Any]
    probabilidad_historica_mensual: List[Dict[str, Any]]
    pronostico_arima: Dict[str, Any]
    nota: str

    class Config:
        schema_extra = {
            "example": {
                "estado": "exito",
                "parametros_entrada": {
                    "ciudad": "Santa Cruz de la Sierra, Bolivia",
                    "evento": "Halloween",
                    "condicion_texto": "llueva",
                    "año_objetivo": 2025
                },
                "ubicacion_geocodificada": {
                    "latitud": -17.8146,
                    "longitud": -63.1561,
                    "nombre_completo": "Santa Cruz de la Sierra, Bolivia"
                },
                "probabilidad_historica_evento": {
                    "probabilidad_porcentaje": 45.5,
                    "dias_favorables": 91,
                    "dias_totales": 200
                },
                "pronostico_arima": {
                    "rmse_evaluacion": 2.08,
                    "condicion_cumplida_en_pronostico": True
                }
            }
        }
