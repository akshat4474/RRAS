@echo off
setlocal
for /f "tokens=*" %%a in ('type .env ^| findstr /v "^#"') do set %%a
streamlit run app/test.py
