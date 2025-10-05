"""
Modelos Pydantic para validación de requests
"""

from pydantic import BaseModel, Field
from typing import Optional


class AnalisisRequest(BaseModel):
    """Request para análisis climático completo"""
    ciudad: str = Field(
        ...,
        description="Nombre de la ciudad (ej. 'Santa Cruz de la Sierra, Bolivia')",
        example="Santa Cruz de la Sierra, Bolivia"
    )
    evento: str = Field(
        ...,
        description="Nombre del evento o fecha (ej. 'Halloween' o '31/10/2025')",
        example="Halloween"
    )
    condicion_texto: str = Field(
        ...,
        description="Condición climática a evaluar (ej. 'llueva', 'caluroso', 'frío')",
        example="llueva"
    )
    año_objetivo: int = Field(
        default=2025,
        description="Año para el cual generar el pronóstico",
        ge=2025,
        le=2030,
        example=2025
    )
    visualizar: bool = Field(
        default=False,
        description="No aplica en API (reservado para versión notebook)"
    )

    class Config:
        schema_extra = {
            "example": {
                "ciudad": "Santa Cruz de la Sierra, Bolivia",
                "evento": "Halloween",
                "condicion_texto": "llueva",
                "año_objetivo": 2025,
                "visualizar": False
            }
        }


class EventoInfo(BaseModel):
    """Información de un evento disponible"""
    nombre: str
    mes: int
    dia: int


class CondicionInfo(BaseModel):
    """Información de una condición climática disponible"""
    nombre: str
    metrica: str
    umbral: float
    condicion: str
    descripcion: str
