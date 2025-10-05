"""
Cliente para interactuar con la API POWER de la NASA
"""

import requests
from datetime import datetime
from typing import Dict, List, Optional


class NASAPowerAPI:
    """
    Cliente para interactuar con la API POWER de la NASA.
    Implementa consultas históricas multi-año y manejo de errores robusto.
    """

    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

    def __init__(self):
        """Inicializa el cliente de la API."""
        self.parametros_base = [
            "T2M_MAX",      # Temperatura máxima a 2m
            "T2M_MIN",      # Temperatura mínima a 2m
            "T2M",          # Temperatura promedio a 2m
            "WS10M_MAX",    # Velocidad máxima del viento a 10m
            "PRECTOTCORR",  # Precipitación total corregida
            "ALLSKY_SFC_SW_DWN"  # Radiación solar
        ]

    def convertir_fecha_api(self, fecha_str: str) -> str:
        """
        Convierte fecha de YYYY-MM-DD a YYYYMMDD (formato NASA POWER).

        Args:
            fecha_str: Fecha en formato YYYY-MM-DD

        Returns:
            Fecha en formato YYYYMMDD
        """
        fecha = datetime.strptime(fecha_str, "%Y-%m-%d")
        return fecha.strftime("%Y%m%d")

    def consultar_historico_multianio(
        self,
        latitud: float,
        longitud: float,
        fecha_inicio: str,
        fecha_fin: str,
        años_historicos: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Consulta datos históricos para múltiples años.

        Args:
            latitud: Latitud del punto de interés
            longitud: Longitud del punto de interés
            fecha_inicio: Fecha de inicio (YYYY-MM-DD)
            fecha_fin: Fecha de fin (YYYY-MM-DD)
            años_historicos: Lista de años a consultar (default: 2005-2024)

        Returns:
            Lista de respuestas JSON de la API
        """
        if años_historicos is None:
            años_historicos = list(range(2005, 2025))  # 20 años de datos

        # Extraer mes y día de las fechas originales
        fecha_obj_inicio = datetime.strptime(fecha_inicio, "%Y-%m-%d")
        fecha_obj_fin = datetime.strptime(fecha_fin, "%Y-%m-%d")
        mes_inicio, dia_inicio = fecha_obj_inicio.month, fecha_obj_inicio.day
        mes_fin, dia_fin = fecha_obj_fin.month, fecha_obj_fin.day

        resultados = []

        print(f"🔄 Consultando datos históricos para {len(años_historicos)} años...")

        for año in años_historicos:
            try:
                # Reconstruir fechas para este año
                fecha_inicio_año = datetime(año, mes_inicio, dia_inicio)
                fecha_fin_año = datetime(año, mes_fin, dia_fin)

                # Convertir a formato API
                start_api = fecha_inicio_año.strftime("%Y%m%d")
                end_api = fecha_fin_año.strftime("%Y%m%d")

                # Construir URL de consulta
                params = {
                    "parameters": ",".join(self.parametros_base),
                    "community": "RE",
                    "longitude": longitud,
                    "latitude": latitud,
                    "start": start_api,
                    "end": end_api,
                    "format": "JSON"
                }

                # Realizar petición
                response = requests.get(self.BASE_URL, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    resultados.append(data)
                    print(f"  ✓ Año {año}: {len(data['properties']['parameter']['T2M'])} días obtenidos")
                else:
                    print(f"  ✗ Año {año}: Error {response.status_code}")

            except Exception as e:
                print(f"  ✗ Año {año}: {str(e)}")
                continue

        print(f"✅ Consulta completada: {len(resultados)} años exitosos\n")
        return resultados
