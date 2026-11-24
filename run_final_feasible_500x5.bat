@echo off
REM ALTERNATIVA COMPUTACIONAL PREDEFINIDA (O1): 500 usuarios x 5 semillas.
REM Documentar en PI13 la desviacion respecto del plan original 1000 x 5.
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
python -m unittest discover -s tests -v
if errorlevel 1 exit /b 1
python run_multiseed.py --users 500 --seeds 42 43 44 45 46 --tuning full --bootstrap 1000 --output outputs_final_500x5 --resume
if errorlevel 1 exit /b 1
python analyze_pi14.py --input outputs_final_500x5 --output outputs_final_500x5/analysis_pi14
pause
