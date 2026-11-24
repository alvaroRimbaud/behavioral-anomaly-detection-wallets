@echo off
set "DATA=data\external\MoMTSim_20240722202413_1000_dataset.csv"
if not exist "%DATA%" (
  echo CSV original de MoMTSim no encontrado: %DATA%
  echo Descargar version 2 desde https://data.mendeley.com/datasets/zhj366m53p/2
  echo NO usar un dataset equilibrado o generado por T12 como evidencia externa.
  exit /b 2
)
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
python run_external_validation.py --dataset momtsim --data "%DATA%" --tuning quick --seed 42 --bootstrap 1000 --min-warmup 5 --user-fraction 0.05 --output outputs_external_momtsim --confirm-original-source
pause
