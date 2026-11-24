# T12 — Correspondencia explícita entre inyección sintética y representación

**Motivación:** observación O2 (PI13–PI14, 10-11-2026). Este cuadro no pretende justificar la independencia entre datos y features: **revela el acoplamiento metodológico** que podría inflar artificialmente el resultado raw → híbrido. La evidencia del simulador sirve para detectar anomalías *definidas en ese entorno*; la prueba de transferibilidad requiere un generador independiente.

## Diseño y amenaza por construcción

- Se generan perfiles por usuario y 30 transacciones legítimas de *warm-up*. Los estadísticos de normalidad `B_u` quedan fijos para train, validation y test. El modelo global no conoce `user_id` como predictor directo; las features son calculadas respecto a `B_u`.
- Para las anomalías ATO en validación/test se eligen dimensiones `A_a` y cantidad `K=|A_a|` (1, 2 o 3). Se asigna una intensidad objetivo, no observada por los detectores. Las intensidades objetivo baja, media y alta se reparten casi equitativamente **antes de evaluar resultados**, dentro de cada fase. La intensidad realmente alcanzada `D` se calcula de las alteraciones; su distribución no se fuerza artificialmente a alcanzar cuotas.
- La prevalencia de anomalía del 2 % y las distribuciones de arquetipos, dispositivos y ubicaciones son **supuestos de simulación**, no tasas verificadas para Yape/Plin.

| Dimensión perturbada | Regla concreta del generador (`src/simulator.py`) | Feature individual que puede captarla (`src/features.py`) | Riesgo específico de circularidad |
|---|---|---|---|
| Monto | Multiplicador exponencial respecto a la mediana del warm-up; ocasional reducción del monto | `amount_deviation_user`: desviación robusta mediana/MAD normalizada y truncada | **Muy alto:** inyección y feature se definen con la misma mediana/MAD |
| Horario | Desplazamiento de 1 a 12 horas respecto a la hora modal del warm-up | `hour_novelty_user`: distancia circular a la hora modal/12 | **Muy alto:** misma hora modal determina inyección y feature |
| Frecuencia | Modifica el intervalo respecto al último evento y gap mediano histórico; deja eventos previos intactos | `frequency_deviation_user`, además `time_since_last_tx` y conteos de 1/24 h | Alto; reglas compartidas de intervalo, y potenciales efectos temporales secundarios |
| Destinatario | Destinatario nuevo cuando intensidad objetivo ≥ 0,67, o destinatario previo según rareza | `recipient_novelty_user`: 1 si no observado; en otro caso 1/(1+frecuencia histórica) | **Muy alto:** se inyecta directamente el mismo tipo de novedad |
| Dispositivo | Dispositivo nuevo cuando intensidad objetivo ≥ 0,67; alternativa habitual en otros casos | `device_novelty_user` con misma regla de novedad | **Muy alto:** coincidencia directa del concepto |
| Ubicación | Zona de distrito no vista cuando intensidad objetivo ≥ 0,67; alternativa habitual en otros casos | `location_novelty_user` basada en la lista de warm-up | **Muy alto:** mismas zonas y frecuencias de referencia |

Los scores de los modelos no contienen `is_anomaly`, `D`, `K`, `changed_dimensions` ni `target_intensity_band`; pero **la exclusión de etiquetas no elimina por sí sola esta circularidad de diseño**. La ablación dentro del simulador estima un efecto condicionado a sus mecanismos, no una mejora generalizable a billeteras productivas.

## Estrategia de mitigación

1. Informar la tabla anterior de forma explícita en método y discusión, sin esconder los acoplamientos.
2. Fijar *a priori* la política de bandas objetivo para impedir que un estrato sea seleccionado mirando los scores. Reportar además distribución de D realmente alcanzado y `n` por banda; no inferir D50 si hay menos de 10 positivos en alguna de las dos bandas usadas para interpolación.
3. Realizar un **challenge de generador independiente** sobre la versión original de MoMTSim (preferida) o PaySim, sin invocar `src/simulator.py`, conservando las etiquetas de fraude de origen para validación y test únicamente. Esto no equivale a validación con datos reales.
4. En datasets externos se compararán solo variables observables: 6 raw/contextuales y 4 individuales (monto, hora, frecuencia y destinatario). **No inventar** `device_id`, zona ni sus respectivas features faltantes. Comparar raw vs híbrida observable *dentro del mismo dataset*, no equiparar directamente sus AP con el experimento de 12 variables.
5. Reportar todas las mejoras, ausencias de mejora, bajos recalls y pérdidas de cobertura de usuarios. Si el efecto no se reproduce, presentarlo como límite de validez externa, no corregir el diseño retrospectivamente para mejorar las cifras.

## Trazabilidad

- Fuente de código: `src/simulator.py`: `_mutate_anomaly`, `_inject_anomalies`; `src/features.py`: `build_features`.
- Instrumentos: `intensity_target_support.csv`, `recall_vs_D.csv`, `d50_estimates.csv`, `metrics_global.csv`, `paired_bootstrap_pr_auc.csv`.
- Evidencia complementaria: `run_external_validation.py`, `external_metrics.csv`, `external_cluster_bootstrap.csv`, `external_run_summary.json`.
- Observación del asesor: *Revisión PI13 y PI14*, T12, O2, 10 de noviembre de 2026.
