@echo off
setlocal

REM Load environment variables from .env file
for /f "tokens=*" %%a in ('type .env ^| findstr /v "^#"') do set %%a

REM Set default port if not set
if "%API_PORT%"=="" set API_PORT=8000

REM Activate virtual environment if exists
if exist "%~dp0..\venv\Scripts\activate.bat" call "%~dp0..\venv\Scripts\activate.bat"

REM Run Uvicorn server
uvicorn api.main:app --reload --port %API_PORT%