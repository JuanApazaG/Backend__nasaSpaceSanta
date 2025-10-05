# NASA SPACE APPS CHALLENGE 2025

"¿Qué significa la 'P' en 'Clima'? ¡Probabilidad!"

Sistema de Análisis Climático Probabilístico con NASA POWER API
================================================================

Este repositorio contiene una API REST construida con FastAPI que calcula la
probabilidad histórica de condiciones climáticas (ej. "llueva", "caluroso") y
genera pronósticos usando modelos ARIMA. El objetivo es ofrecer una herramienta
ligera y reproducible para evaluar riesgos climáticos alrededor de eventos
locales (fiestas, desfiles, festivales, etc.).

Contenido principal
-------------------

- `main.py` - Aplicación FastAPI (endpoints: `/api/v1/analisis`, `/api/v1/eventos`, `/api/v1/condiciones`, `/health`).
- `models/` - Modelos Pydantic para requests/responses.
- `services/` - Lógica modular: geocodificación, cliente NASA POWER, procesamiento, análisis probabilístico y forecaster ARIMA.
- `requirements.txt` - Dependencias usadas para correr la API.

Características clave
---------------------

- Geocodificación de ciudades usando `geopy` (Nominatim).
- Consulta histórica multi-año a la NASA POWER API para métricas como `T2M`, `PRECTOTCORR`, `WS10M_MAX`, `ALLSKY_SFC_SW_DWN`.
- Unión y limpieza de series temporales en un DataFrame maestro (Pandas).
- Cálculo de probabilidades históricas por frecuencia y por mes.
- Pronóstico con ARIMA (`statsmodels`) y evaluación por RMSE sobre un conjunto de prueba.

Requisitos
---------

- Python 3.11 recomendado (algunas dependencias se compilan y funcionan mejor con 3.11)
- pip

Instalación local (desarrollo)
-------------------------------

1. Clonar el repositorio y entrar en la carpeta:

    ```powershell
    cd D:\Escritorio
    git clone <tu-repo-url>
    cd backend_hackathon
    ```

2. (Opcional) crear y activar un virtualenv

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

3. Instalar dependencias:

    ```powershell
    pip install -r requirements.txt
    ```

4. Ejecutar localmente:

    ```powershell
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

5. Documentación automática:

    - Swagger UI: http://localhost:8000/docs
    - ReDoc:     http://localhost:8000/redoc

Endpoints principales
---------------------

- `POST /api/v1/analisis` - Body: `AnalisisRequest` (ciudad, evento, condicion_texto, año_objetivo)
  - Retorna probabilidad histórica, pronóstico ARIMA, RMSE y metadatos.
- `GET /api/v1/eventos` - Lista de eventos mapeados (p.ej. Halloween, Carnaval).
- `GET /api/v1/condiciones` - Lista de condiciones disponibles y sus parámetros.
- `GET /health` - Health-check simple.

Notas sobre dependencias que pueden dar problemas
-----------------------------------------------

- `pandas`, `numpy`, `statsmodels` y `scikit-learn` tienen componentes compilados. Para
  evitar fallos en el build (especialmente en plataformas con Python 3.13), recomendamos usar
  Python 3.11 o desplegar mediante Docker para controlar la imagen base.

Despliegue (opciones fáciles)
-----------------------------

1) Render (recomendado para simplicidad)
   - Conecta tu repo de GitHub y crea un "Web Service".
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Si la build falla por compilación de paquetes, usa Docker (ver más abajo).

2) Fly.io (buena opción si quieres usar Docker o pequeña latencia global)
   - Instala `flyctl`, `fly launch` y despliega con `flyctl deploy`.
   - Crear un `Dockerfile` (ejemplo incluido en el repo si lo necesitas).

3) Railway (similar a Render)
   - Conecta repo y configura Build/Start commands (igual que Render).

Despliegue con Docker (más robusto)
----------------------------------

Si quieres evitar problemas de compilación, añade un `Dockerfile` y despliega la imagen.

Ejemplo de `Dockerfile` recomendado:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential gfortran libatlas-base-dev --no-install-recommends \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y build-essential gfortran libatlas-base-dev \
    && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
COPY . .
ENV PORT=8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
```

Esto fuerza Python 3.11 y preinstala compiladores necesarios para `pandas/statsmodels`.

Troubleshooting rápido
----------------------

- Error de compilación de pandas en CI (ej. en Render): usa Dockerfile con Python 3.11 o añade `runtime.txt` con `python-3.11.x`.
- Error "Timestamp not JSON serializable": convertimos fechas a strings en `main.py` antes de devolver JSON.
- Si la API recibe errores 500, revisa los logs del servicio (Render/Fly/Railway) y los logs locales.

Contribuir
---------

Si quieres mejorar el proyecto:

- Añade tests unitarios para `services/`.
- Implementa caching (por ejemplo Redis) para no reconsultar la API de NASA con cada petición.
- Añade autenticación y rate-limiting si la API se va a exponer públicamente.

Contacto y licencia
-------------------

Proyecto: NASA Space Apps Challenge 2025 — MIT License

---

Si quieres, ahora creo los archivos `Dockerfile`, `.dockerignore` y `Procfile` y hago un commit con ellos. Dime si quieres que los suba al repo y preparo los pasos exactos para desplegar en Render con Docker.

