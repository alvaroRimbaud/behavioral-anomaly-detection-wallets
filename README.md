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

## Revisión PI13–PI14 del asesor: O1 y O2 (10-11-2026)

**O1 — Multisemilla y recursos:** `run_multiseed.py --resume` comprueba presencia de evidencia completa y coincidencia de seed, muestra, tuning y bootstrap antes de reutilizar cualquier corrida. Genera `scores_test_all_models.csv` para auditoría independiente de las seis combinaciones modelo/representación. La alternativa **500 usuarios × 5 semillas** se define previamente como desviación por recursos y se inicia con `run_final_feasible_500x5.bat`, no se selecciona por sus resultados. Si la escala original es factible, mantener 1000×5. Consultar [protocolo O1](docs/PROTOCOLO_FINAL_O1.md).

**O2 — Riesgo por construcción:** la [matriz explícita](docs/MECANISMOS_SIMULADOR_VS_FEATURES.md) documenta cómo la inyección afecta exactamente las features del baseline y el riesgo de circularidad. Se incluye [guía para contraste en MoMTSim independiente](docs/VALIDACION_EXTERNA_MOMTSIM.md) y `run_external_validation.py`. El challenge usa **las etiquetas originales**, no invoca el simulador T12, separa pasos temporales y utiliza solamente 6 raw + 4 individualizadas observables y, para viabilidad local, una muestra reproducible del 5 % de usuarios del CSV original (sin consultar etiquetas). Es externo respecto del generador, pero sigue siendo un dataset **sintético**. El CSV original se debe descargar de Mendeley; **no se incluye** en el repo y no se atribuyen resultados externos hasta su ejecución verificada.

**Intensidad y D50:** los objetivos de intensidad baja/media/alta se balancean a priori por fase, mientras que la intensidad conseguida D permanece libre y se reporta en `intensity_target_support.csv`. D50 no se estima si no alcanza Recall de 0.5 o las bandas de cruce no contienen al menos diez anomalías cada una. Un `NaN` no se rellena ni se considera fallo del software.

**Rigor estadístico:** `bootstrap_summary.csv` solo consolida diferencias medias entre semillas; **no promedia límites de intervalos de confianza**. `analyze_pi14.py` calcula adicionalmente una prueba exacta bilateral por semilla y explica que con cinco semillas su menor valor p posible es 0.0625. No se declara significancia a 5 % a partir de un test que no puede alcanzarla.

**Comprobar integridad:** `python -m unittest discover -s tests -v`; tests para fuentes externas, exclusión de etiquetas, bandas target, D50 y `--resume`. La prueba sobre CSV sintético de `tests/fixtures/` es solamente **smoke de formato**, no valida O2.

## Integración PI15 — adquisición de fuente independiente (23/11/2026)

Se agrega `download_momtsim_original.py` para consultar metadatos oficiales del depósito Mendeley Data v2 y producir una ficha SHA-256. La validación externa **NO** se considera ejecutada por instalar el script ni por correr pruebas de esquema. Debe obtenerse el CSV original, registrarse su procedencia y ejecutar `run_external_validation.py` con etiquetas originales (véase `docs/VALIDACION_EXTERNA_MOMTSIM.md`). En esta versión, la suite local incluye 10 pruebas metodológicas y de adquisición. Los archivos docentes PI15 y tesis revisada se distribuyen en un paquete aparte y no en el código productivo.
