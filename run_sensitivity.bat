@echo off
REM Analisis de sensibilidad de warm-up y prevalencia sintetica.
python run_sensitivity.py --users 300 --seed 42 --tuning off --bootstrap 200 --output outputs_sensitivity
pause
