"""
Servicio de traducción de entrada del usuario
Convierte lenguaje natural a parámetros técnicos
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
from geopy.geocoders import Nominatim


class UserInputTranslator:
    """
    Clase para traducir entradas en lenguaje natural a parámetros técnicos.
    """

    def __init__(self):
        """Inicializa el traductor con diccionarios de eventos y condiciones."""

        # Diccionario de eventos populares en Latinoamérica
        self.eventos = {
            # Eventos internacionales
            "año nuevo": {"mes": 1, "dia": 1},
            "san valentín": {"mes": 2, "dia": 14},
            "día del trabajo": {"mes": 5, "dia": 1},
            "halloween": {"mes": 10, "dia": 31},
            "navidad": {"mes": 12, "dia": 25},

            # Eventos latinoamericanos
            "carnaval": {"mes": 2, "dia": 28},  # Variable, aproximado
            "semana santa": {"mes": 4, "dia": 15},  # Variable, aproximado
            "día de la independencia": {"mes": 8, "dia": 6},  # Bolivia
            "día del estudiante": {"mes": 9, "dia": 21},
            "día de muertos": {"mes": 11, "dia": 2},
            "san juan": {"mes": 6, "dia": 24},
        }

        # Diccionario de condiciones climáticas
        self.condiciones = {
            "llueva": {
                "metrica": "PRECTOTCORR",
                "umbral": 0.1,
                "condicion": "mayor_que",
                "descripcion": "Precipitación total > 0.1 mm"
            },
            "caluroso": {
                "metrica": "T2M_MAX",
                "umbral": 32,
                "condicion": "mayor_que",
                "descripcion": "Temperatura máxima > 32°C"
            },
            "frío": {
                "metrica": "T2M_MIN",
                "umbral": 5,
                "condicion": "menor_que",
                "descripcion": "Temperatura mínima < 5°C"
            },
            "ventoso": {
                "metrica": "WS10M_MAX",
                "umbral": 20,
                "condicion": "mayor_que",
                "descripcion": "Viento máximo > 20 km/h"
            },
            "soleado": {
                "metrica": "ALLSKY_SFC_SW_DWN",
                "umbral": 20,
                "condicion": "mayor_que",
                "descripcion": "Radiación solar > 20 MJ/m²"
            }
        }

        # Inicializa geocodificador
        self.geolocator = Nominatim(user_agent="nasa_space_apps_2025")

    def geocodificar_ciudad(self, ciudad: str) -> Optional[Dict[str, float]]:
        """
        Convierte el nombre de una ciudad en coordenadas geográficas.

        Args:
            ciudad: Nombre de la ciudad (ej. "Santa Cruz de la Sierra, Bolivia")

        Returns:
            Diccionario con 'latitud' y 'longitud'
        """
        try:
            ubicacion = self.geolocator.geocode(ciudad)
            if ubicacion:
                return {
                    "latitud": round(ubicacion.latitude, 4),
                    "longitud": round(ubicacion.longitude, 4),
                    "nombre_completo": ubicacion.address
                }
            else:
                return None
        except Exception as e:
            print(f"❌ Error en geocodificación: {e}")
            return None

    def interpretar_fecha(self, evento: str, año: int = 2025) -> Optional[Dict[str, str]]:
        """
        Convierte un evento o fecha en un rango de fechas técnico.

        Args:
            evento: Nombre del evento o fecha (ej. "Halloween")
            año: Año para el cual calcular la fecha

        Returns:
            Diccionario con 'fecha_inicio' y 'fecha_fin' en formato YYYY-MM-DD
        """
        evento_lower = evento.lower().strip()

        # Buscar en diccionario de eventos
        if evento_lower in self.eventos:
            evento_data = self.eventos[evento_lower]
            fecha = datetime(año, evento_data["mes"], evento_data["dia"])

            # Rango de ±3 días alrededor del evento
            fecha_inicio = fecha - timedelta(days=3)
            fecha_fin = fecha + timedelta(days=3)

            return {
                "fecha_inicio": fecha_inicio.strftime("%Y-%m-%d"),
                "fecha_fin": fecha_fin.strftime("%Y-%m-%d"),
                "fecha_central": fecha.strftime("%Y-%m-%d")
            }

        # Intentar parsear fecha directa
        try:
            fecha = datetime.strptime(evento, "%d/%m/%Y")
            fecha_inicio = fecha - timedelta(days=3)
            fecha_fin = fecha + timedelta(days=3)
            return {
                "fecha_inicio": fecha_inicio.strftime("%Y-%m-%d"),
                "fecha_fin": fecha_fin.strftime("%Y-%m-%d"),
                "fecha_central": fecha.strftime("%Y-%m-%d")
            }
        except:
            return None

    def mapear_condicion(self, condicion: str) -> Optional[Dict]:
        """
        Mapea una condición en lenguaje natural a parámetros técnicos.

        Args:
            condicion: Descripción de la condición (ej. "llueva", "caluroso")

        Returns:
            Diccionario con parámetros técnicos de la condición
        """
        condicion_lower = condicion.lower().strip()

        # Buscar palabras clave
        for key, params in self.condiciones.items():
            if key in condicion_lower:
                return params

        return None
