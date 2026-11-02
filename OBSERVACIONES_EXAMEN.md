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

## Ajuste posterior a la revisión PI9 + PI10 (PI11 + PI12)

- El bootstrap pareado fue corregido para remuestrear **usuarios completos** (bootstrap por conglomerados), no transacciones individuales. El archivo `paired_bootstrap_pr_auc.csv` registra `bootstrap_unit=user` y `n_clusters`.
- Se añadió `d50_estimates.csv`, donde D50 se define como la intensidad D a partir de la cual la curva Recall-vs-D alcanza 0.50, con interpolación lineal entre bins adyacentes. Este indicador se utilizará de forma comparativa entre IF, Autoencoder y LOF; un D50 menor implica sensibilidad frente a anomalías menos intensas.
- `run_multiseed.py` consolida D50 por semilla en `d50_all_seeds.csv` y `d50_summary.csv`.
- La corrida definitiva continúa planificada con 1 000 usuarios x 5 semillas. La sensibilidad del simulador permanece pendiente de ejecución definitiva; `run_sensitivity.py` contiene los escenarios de warm-up 20/30/40 y prevalencia 1/2/5 %.
