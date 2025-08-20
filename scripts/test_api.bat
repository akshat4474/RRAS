@echo off
REM ---- RRAS API smoke tests ----
setlocal

REM If running from project root, ensure core imports work
set PYTHONPATH=%CD%

echo.
echo [1/3] /allocate/run
curl -s -X POST http://127.0.0.1:8000/allocate/run -F "areas=@data/mini/areas.csv"

echo.
echo [2/3] /route/run
curl -s -X POST http://127.0.0.1:8000/route/run -F "roads=@data/mini/roads.csv"-F "src=D1" -F "dst=A2"

echo.
echo [3/3] /plan/run
curl -s -X POST http://127.0.0.1:8000/plan/run -F "areas=@data/mini/areas.csv" -F "depots=@data/mini/depots.csv" -F "roads=@data/mini/roads.csv"

echo.
echo Done.
endlocal
