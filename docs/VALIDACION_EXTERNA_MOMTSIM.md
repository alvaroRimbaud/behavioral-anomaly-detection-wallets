# T12 — Validación con generador independiente (O2)

## Propósito y límites

Se evalúa si la mejora de la representación por baseline personal obtenida en el simulador T12 se mantiene en un conjunto de transacciones que **no fue generado por T12**. Se usará preferentemente el CSV **original y completo** de MoMTSim versión 2, con etiquetas originales. Es todavía un conjunto **sintético**, no información real de Yape/Plin ni validación operativa.

**Fuente:** Azamuke, D., Katarahweire, M., & Bainomugisha, E. (2025). *A labeled synthetic mobile money transaction dataset*. Data in Brief, 60, 111534, DOI: 10.1016/j.dib.2025.111534. Depósito: [Mendeley Data v2](https://data.mendeley.com/datasets/zhj366m53p/2), DOI: 10.17632/zhj366m53p.2. Mendeley declara licencia CC BY 4.0 (verificar versión y términos al descargar).

**Advertencia:** los CSV de terceros de 30 000/50 000 filas encontrados en GitHub pueden ser muestras estratificadas o reequilibradas y **no constituyen el dataset original**. No deben usarse como evidencia concluyente de prevalencia, PR-AUC o transferibilidad. Para empezar, la rutina cuenta con un *smoke test* sintético **propio** de esquema, claramente marcado como tal.

## Procedimiento fijo (separación temporal sin etiqueta-oráculo)

1. Descargar MoMTSim original desde Mendeley y conservar nombre, versión, licencia, tamaño de archivo y SHA-256.
2. Usar `step` como horas simuladas. Asignar fases por **pasos enteros completos**, en el orden temporal: 20 % warm-up, 50 % train, 15 % validation y 15 % test. No mezclar un mismo `step` entre train/validation/test.
3. Por límite de recursos, seleccionar **antes de observar etiquetas o scores** un 5 % determinista de iniciadores mediante SHA-256 del ID, reteniendo toda su historia a través de las fases. Documentar la fracción exacta; no confundir esta muestra aleatoria por usuario con el dataset completo ni con una muestra estratificada por fraude. Mantener solo iniciadores con **al menos 5 transacciones previas de warm-up**, con independencia de la etiqueta. Construir mediana/MAD de monto, hora modal, intervalo intertransaccional y frecuencias de destinatario sobre **todas** las observaciones de warm-up elegibles (posibles fraudes incluidos, porque no se consulta `isFraud`). Reportar cobertura de usuarios y transacciones.
4. Extraer seis features raw: monto estandarizado con train, seno/coseno de la hora (`step % 24`), tiempo desde transacción anterior, conteos de 1 y 24 horas con pasado estricto. Extraer cuatro features individuales: desviación de monto, novedad horaria, desviación de frecuencia y novedad de destinatario. **No están disponibles dispositivo ni ubicación**, por lo que el challenge utiliza 6 vs 10 y no replica íntegramente las 12 features del simulador T12.
5. Entrenar IF/AE/LOF globales sobre train sin utilizar etiquetas. El training externo **puede contener fraude según sus etiquetas originales**; no se elimina con `isFraud`, por lo que la condición es detección no supervisada con contaminación natural, diferente al train limpio por diseño de T12. También el holdout interno de early stopping del AE es **no etiquetado y potencialmente contaminado**: nunca debe describirse como normal-only en el experimento externo.
6. Validación utiliza las etiquetas **originales** exclusivamente para búsqueda de hiperparámetros y umbral F1 de referencia. Se calculan AP/PR-AUC y Precision@0.5 %, 1 %, 2 %, 5 % en test, con ground truth sin modificaciones. El bootstrap por usuario estima incertidumbre de las diferencias híbrido–raw.
7. Reportar: datos de origen (SHA-256), prevalencia real del dataset de origen por partición, cobertura del historial, métricas completas de los tres detectores y ΔAP. No mezclar esta prueba con la validación interna ni cambiar parámetros después de mirar test.

`isFraud` no se usa para depurar transacciones de training/warm-up, construir baselines ni calcular features. La etiqueta solo interviene en selección de configuración en validation y evaluación en test. El resultado puede diferir por definición distinta de fraude y por ausencia de contexto de dispositivo/ubicación.

## Ejecutar en Windows / VS Code

```powershell
# Guardar el CSV descargado directamente de Mendeley, p. ej.:
# data/external/MoMTSim_20240722202413_1000_dataset.csv
python -m pip install -r requirements.txt
python run_external_validation.py --dataset momtsim `
  --data data/external/MoMTSim_20240722202413_1000_dataset.csv `
  --tuning quick --seed 42 --bootstrap 1000 --min-warmup 5 --user-fraction 0.05 `
  --output outputs_external_momtsim --confirm-original-source
```

La selección por usuario se ejecuta sobre el CSV ORIGINAL completo y usa una función hash fija independiente de `isFraud`; el archivo conserva su hash SHA-256. El resultado se refiere estrictamente al subconjunto de usuarios seleccionados, no al total de transacciones del depósito.

El flag `--confirm-original-source` representa una **declaración de procedencia del ejecutor**: usarlo solo después de comprobar que es el CSV original, no un archivo de T12 ni una muestra reequilibrada. El programa conserva hash SHA-256 para auditar. Si el CSV completo no está disponible, **no declarar completada O2**.

**PaySim alternativo:** si MoMTSim no se obtiene, usar `--dataset paysim` con [PaySim original](https://www.kaggle.com/datasets/ealaxi/paysim1). Muchos iniciadores de PaySim presentan escaso historial; el script puede retener pocos usuarios o abortar, hecho que debe reportarse como limitación y no subsanarse inventando registros.

## Criterios de decisión e interpretación

- **Evidencia compatible con transferibilidad:** ΔAP híbrido–raw > 0 con intervalo por usuarios que no abarque 0 para al menos los modelos principales, interpretado dentro de la cobertura conseguida. No trasladar a plataformas reales.
- **Evidencia no compatible:** ΔAP ≤ 0 o intervalo que abarque 0. Es un resultado legítimo: delimita el alcance del hallazgo en T12.
- **Sin evidencia externa suficiente:** archivo original no obtenido, muy pocos usuarios elegibles o validation/test sin suficientes positivos; informar motivo, cobertura y no cerrar la observación.
- Comparar AP contra prevalencia observada de cada test: una AP de 0.2 no significa lo mismo con 0.1 % que con 50 % de fraude.

## Evidencias a conservar

`external_run_summary.json` (incluye SHA), `external_metrics.csv`, `external_precision_at_k.csv`, `external_hyperparameter_search.csv`, `external_cluster_bootstrap.csv` y comando completo. No versionar el CSV íntegro en GitHub; documentar fuente y hash y subir solo salidas agregadas.

## Adquisición verificable del CSV original (PI15)

Primero, **no** use el archivo de pruebas `tests/fixtures` ni una copia de procedencia desconocida. El repositorio académico de origen declara versión 2 y dos CSV (alrededor de 149 MB y 349 MB). Se incorpora `download_momtsim_original.py` para consultar metadatos del editor, descargar si la API lo permite y guardar SHA-256:

```powershell
python download_momtsim_original.py --catalog
python download_momtsim_original.py --download --output data/external
# Si la API del editor bloquea la descarga: hacerlo manualmente desde
# https://data.mendeley.com/datasets/zhj366m53p/2
python download_momtsim_original.py --inspect-local data/external/ARCHIVO_ORIGINAL.csv --output data/external
```

El archivo `momtsim_provenance.json` es una ficha de integridad; **no prueba por sí solo** que un CSV cargado manualmente es el original. El tesista debe cotejar su nombre, tamaño, licencia, página oficial y, si está disponible, checksum del editor. La validación se considera ejecutada solamente cuando `run_external_validation.py` genere las métricas sobre el archivo confirmado y se conserve el comando completo. La imposibilidad de descargar desde otro entorno no equivale a un resultado negativo de O2.
