# T12 — Protocolo de verificación PI13/PI14

## Integridad del experimento

- Se conserva el baseline congelado del warm-up legítimo.
- Los detectores no reciben `is_anomaly`, `anomaly_intensity` ni atributos de ground truth.
- La selección de hiperparámetros y del threshold ocurre en validation, nunca sobre test.
- El early stopping del Autoencoder utiliza únicamente holdout temporal normal interno.
- Bootstrap pareado remuestrea usuarios completos (`bootstrap_unit=user`).
- LOF es línea base formal y se evalúa con iguales splits y feature sets.

## Ejecución final requerida (equipo del tesista)

```bash
python -m unittest discover -s tests -v
python run_multiseed.py --users 1000 --seeds 42 43 44 45 46 --tuning full --bootstrap 1000 --output outputs_final_5seeds --resume
python run_sensitivity.py --users 300 --seed 42 --tuning quick --bootstrap 200 --output outputs_sensitivity
python analyze_pi14.py --input outputs_final_5seeds --output outputs_final_5seeds/analysis_pi14
```

`--resume` permite completar corridas prolongadas sin recalcular semillas terminadas. Los resultados finales requieren que existan las cinco carpetas completas. No se deben describir corridas piloto como validación definitiva.

## Versionado

Incluya scripts, configuraciones, informe de entorno, outputs compactos y hashes en Git; evite publicar datasets masivos. El vídeo corresponde al tesista.
