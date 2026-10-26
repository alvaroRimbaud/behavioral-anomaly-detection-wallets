# Evidencia reproducible del levantamiento

Esta carpeta contiene únicamente resultados compactos y verificables que sí se versionan. Los datasets completos y outputs pesados continúan excluidos por `.gitignore`.

## `notebook_seed42_120users/`

Corrida ejecutada desde `notebooks/T12_Demo_Artefacto.ipynb` después de incorporar las observaciones del examen:

- seed: 42;
- 120 usuarios / 12 000 transacciones;
- tuning: perfil `quick`;
- bootstrap pareado: 200 remuestreos;
- propósito: prueba reproducible del pipeline actualizado, **no resultado final de tesis**.

La evidencia incluye métricas, hiperparámetros seleccionados, Precision@k, curva Recall vs D, bootstrap pareado, supuestos del simulador y resumen de la corrida.
