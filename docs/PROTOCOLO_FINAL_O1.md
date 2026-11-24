# T12 — Protocolo de cierre O1 fijado por límites de recursos

**Observación del asesor (10-11-2026):** O1 exige más de una semilla; admite explícitamente `1000 × 3` o `500 × 5` si se justifica la limitación de cómputo **antes** de mirar resultados definitivos.

**Plan originalmente propuesto:** 1000 usuarios, 100 transacciones por usuario, cinco semillas 42–46, tuning `full`, 1000 réplicas de bootstrap por usuario.

**Alternativa computacional predeterminada, SOLO si falla el plan original:** **500 usuarios × 5 semillas** (42–46), 100 transacciones por usuario, 2 % de anomalías en validation/test, tuning `full` y 1000 réplicas. Motivo exclusivo: disponibilidad computacional; no se ha escogido tras comparar PR-AUC, y no se ajustará para favorecer a ningún modelo. Si se adopta, registrar en PI13 como desviación y en la tesis como limitación de tamaño de muestra. La comparación debe conservar los mismos 500 usuarios en los tres detectores, seis vs doce features y fases 30/49/10/11.

Antes de una corrida final, dejar asentados fecha, commit, hardware, versión de Python, dependencias y tamaño elegido en `protocolo_ejecutado.json` (registro del propio equipo). El script `run_multiseed.py` verifica que la evidencia completa corresponda a **la misma combinación** de seed/usuarios/tuning/bootstrap y no reutiliza silenciosamente una corrida incompatible con `--resume`.

## Comandos para una semilla por sesión, reanudables

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
python -m unittest discover -s tests -v
# Plan original (una semilla):
python run_multiseed.py --users 1000 --seeds 42 --tuning full --bootstrap 1000 --output outputs_final_5seeds --resume
# Repetir para 43, 44, 45, 46; luego consolidar con las cinco semillas.
python run_multiseed.py --users 1000 --seeds 42 43 44 45 46 --tuning full --bootstrap 1000 --output outputs_final_5seeds --resume
```

Si se activa la alternativa por recursos, ejecutar **todos los comandos** con `--users 500` y **otra carpeta** de salida `outputs_final_500x5` para no mezclar evidencias.

## Reporte

Para cada detector/representación: media, DE de AP/PR-AUC en cinco semillas, Precision@1 %, Recall@1 %, estabilidad de threshold F1 (criterio secundario), validación de train limpio y conteos reales por banda D alcanzada. Bootstrap pareado por usuario para comparaciones dentro de cada semilla. No promediar los extremos de intervalos bootstrap por semilla y llamarlos IC agregado.

**Limitación estadística expresa:** con solo cinco semillas, una prueba exacta bilateral de inversión de signos tiene mínimo p=2/32=0.0625. Por tanto, **no puede alcanzar 0.05**, aunque todas las semillas favorezcan la representación híbrida. Reportar efectos, incertidumbre por usuario, consistencia direccional y esta limitación; no inventar un p<0.05 ni aceptar automáticamente la hipótesis.

## D50

El indicador D50 es **exploratorio**. Se registra solo si la curva Recall frente a D tiene cruce del 50 % con al menos diez anomalías en cada banda adyacente utilizada para interpolar. Si no alcanza Recall de 0.50 o faltan casos, D50 = no estimable. Informar siempre la cantidad de anomalías reales por banda objetivo y por banda D alcanzada. La inyección equilibrada entre bandas objetivo se predefine antes de observar cualquier score.
