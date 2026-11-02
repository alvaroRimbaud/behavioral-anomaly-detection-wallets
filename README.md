# T12 — Artefacto experimental reproducible

**Tesis:** *Detección de anomalías conductuales en transacciones de billeteras digitales mediante modelos de aprendizaje automático no supervisado*.

Este repositorio implementa el pipeline experimental de la tesis y refleja el levantamiento de las observaciones del examen parcial. Isolation Forest (IF) y Autoencoder (AE) permanecen como modelos centrales; Local Outlier Factor (LOF) se incorpora como **baseline formal** bajo el mismo protocolo experimental.

## Flujo experimental

1. Simulación reproducible de perfiles y transacciones.
2. Warm-up legítimo y construcción de baseline individual congelado.
3. Ingeniería de 12 features: 6 raw/contextuales + 6 individualizadas.
4. Split temporal: train normal → validation → test.
5. Holdout temporal **normal-only** dentro de train para early stopping del AE.
6. Búsqueda de hiperparámetros usando exclusivamente validation para selección experimental.
7. Anomaly score por IF, AE y LOF.
8. Threshold F1 seleccionado en validation como política de referencia; test permanece congelado.
9. Política adicional por presupuesto de alertas: top 0.5 %, 1 %, 2 % y 5 %, con Precision@k y Recall@k.
10. PR-AUC, Precision, Recall, F1, FP/FN, Recall frente a D, análisis por dimensión y bootstrap pareado.

## Cambios por observaciones del examen

- **Obs. 1 — LOF:** deja de ser “auxiliar” y se evalúa como baseline formal, incluyendo bootstrap AE–LOF e IF–LOF.
- **Obs. 2 — Evidencia:** el notebook queda preparado para ejecución reproducible y se incluye `evidence/` con una corrida smoke verificable. Los commits deben reflejar el proceso real; no se retrofechan commits.
- **Obs. 3 — Tesis:** se resuelve en el documento de tesis por el tesista; no se altera desde este repositorio.
- **Obs. 4 — Hipótesis/OE:** el código genera evidencia de ablación raw vs híbrido y bootstrap que permite contrastar la hipótesis de aporte de features individualizadas.
- **Obs. 5 — Estratos:** D se mantiene continuo y se genera `recall_vs_D.csv` + `recall_vs_D.png`; la corrida final aumenta el número de casos con 1 000 usuarios × 5 semillas.
- **Obs. 6 — Hiperparámetros/early stopping/simulador:** búsqueda real de hiperparámetros; early stopping sobre holdout normal independiente de `train_fit`; supuestos del simulador quedan exportados en `simulator_assumptions.csv` y existe `run_sensitivity.py`.
- **Obs. 7 — Threshold:** se conserva max-F1 en validation como referencia, se reporta estabilidad multi-semilla y se agrega política top-k/Precision@k.

## Ejecución de demostración

Windows:

```bat
run_quick_demo.bat
```

Corrida S8 actualizada (300 usuarios):

```bat
run_demo.bat
```

## Experimento final planificado

Ejecuta cinco semillas, 1 000 usuarios por semilla, tuning completo y bootstrap pareado de 1 000 remuestreos:

```bat
run_final_experiment.bat
```

Equivalente:

```bash
python run_multiseed.py --seeds 42 43 44 45 46 --users 1000 --tuning full --bootstrap 1000
```

El consolidado genera `metrics_and_threshold_stability.csv`, `precision_at_k_summary.csv` y `bootstrap_summary.csv`.

## Sensibilidad del simulador

```bat
run_sensitivity.bat
```

Evalúa configuraciones de warm-up (20/30/40) y prevalencia sintética (1/2/5 %). Los valores se documentan como supuestos experimentales cuando no representan parámetros empíricos de una billetera real.

## Archivos principales de evidencia por corrida

- `metrics_global.csv`
- `hyperparameter_search.csv`
- `selected_hyperparameters.csv`
- `threshold_candidates_validation.csv`
- `precision_at_k.csv`
- `recall_vs_D.csv` y `recall_vs_D.png`
- `paired_bootstrap_pr_auc.csv`
- `simulator_assumptions.csv`
- `reporte_demo.html`

## Reglas metodológicas preservadas

- Ground truth **no entra a X_train** ni al fit de IF/AE/LOF.
- Train contiene normalidad por diseño; las anomalías se generan solo en validation/test.
- El baseline individual queda congelado al finalizar warm-up.
- El conjunto test no selecciona hiperparámetros ni threshold.
- Un anomaly score alto se interpreta de manera uniforme como mayor anomalía, aunque las escalas entre modelos no se comparan directamente.
- Los resultados sintéticos no se presentan como desempeño productivo en Yape, Plin u otra plataforma real.

### Ajustes para PI11 + PI12

La validación definitiva usa bootstrap pareado **por usuario** para respetar la dependencia entre transacciones de una misma persona. Además, el pipeline exporta `d50_estimates.csv`: D50 es el punto de intensidad conductual en el que Recall alcanza 0.50 en la curva Recall-vs-D. `run_multiseed.py` consolida la estabilidad de D50, thresholds e hiperparámetros entre semillas.
