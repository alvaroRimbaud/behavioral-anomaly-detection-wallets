# T12 — Trazabilidad del levantamiento de observaciones del examen parcial

Este archivo vincula las observaciones del docente con cambios verificables del artefacto. Los resultados definitivos de cinco semillas no se declaran ejecutados hasta correr `run_final_experiment.bat`.

| N.º | Observación | Cambio verificable en el artefacto | Estado del artefacto |
|---|---|---|---|
| 1 | LOF iguala/supera modelos centrales | LOF pasa a `LOF baseline`, usa los mismos splits/feature sets, tuning en validation y bootstrap pareado AE–LOF / IF–LOF | Implementado |
| 2 | Evidencia de proceso | Notebook ejecutado; `evidence/` contiene resultados compactos; se agregaron commits reales de esta sesión sin retrofechar | Implementado; falta publicar/push por el tesista |
| 3 | Tesis desactualizada | No corresponde al código; debe actualizarse en el documento de tesis | Pendiente a cargo del tesista |
| 4 | Hipótesis y objetivos | El artefacto genera ablación raw vs híbrido y CI bootstrap de ΔPR-AUC para evidencia contrastable | Implementación de evidencia lista; redacción de hipótesis corresponde a tesis |
| 5 | Pocos casos por estrato | `recall_vs_D.csv/png` conserva D continuo; final configurado a 1 000 usuarios × 5 semillas | Implementado; corrida final pendiente |
| 6 | Hiperparámetros, early stopping y simulador | Grid real IF/AE/LOF; AE usa holdout temporal normal-only; `simulator_assumptions.csv`; `run_sensitivity.py` | Implementado; sensibilidad definitiva pendiente de ejecutar |
| 7 | Umbral de decisión | Max-F1 en validation se conserva como referencia; `precision_at_k.csv`; multi-seed consolida estabilidad del threshold | Implementado; corrida final pendiente |

## Regla de cierre

Una observación que depende de cinco semillas, sensibilidad o resultados definitivos solo puede marcarse como **levantada** después de ejecutar la corrida correspondiente y registrar la evidencia resultante. El código preparado no debe confundirse con un resultado ya ejecutado.
