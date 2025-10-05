@echo off
echo ========================================
echo NASA Climate Probability API
echo ========================================
echo.
echo Instalando dependencias...
pip install -r requirements.txt
echo.
echo Iniciando servidor FastAPI...
echo La API estara disponible en: http://localhost:8000
echo Documentacion Swagger: http://localhost:8000/docs
echo.
uvicorn main:app --reload --host 0.0.0.0 --port 8000
