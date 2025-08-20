@echo off
setlocal
for /f "tokens=*" %%a in ('type .env ^| findstr /v "^#"') do set %%a
streamlit run app/app.py
