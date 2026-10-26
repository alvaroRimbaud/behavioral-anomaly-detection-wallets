@echo off
REM Experimento final: 5 semillas, 1000 usuarios, tuning completo y bootstrap 1000.
python run_multiseed.py --seeds 42 43 44 45 46 --users 1000 --tuning full --bootstrap 1000 --output outputs_final_5seeds
pause
