# T12 — Artefacto inicial S8

**Tesis:** *Detección de anomalías conductuales en transacciones de billeteras digitales mediante modelos de aprendizaje automático no supervisado*.

Este repositorio contiene una construcción inicial reproducible del pipeline de tesis. Está diseñado para la demostración del examen parcial y para servir como base de la implementación final.

## Qué demuestra

1. Generación de transacciones sintéticas y perfiles heterogéneos.
2. Warm-up legítimo por usuario.
3. Línea base conductual **congelada** después del warm-up.
4. Doce características: 6 absolutas/contextuales + 6 individualizadas.
5. Train temporal **sin anomalías por diseño**.
6. Anomalías ATO controladas solo en validation/test, con intensidad continua D y amplitud K.
7. Isolation Forest y Autoencoder como modelos principales.
8. LOF como **baseline auxiliar para la demo**, sin modificar el objetivo central de la tesis.
9. Selección de umbral en validation y evaluación final en test.
10. Reporte HTML con métricas reales de la corrida, recall por intensidad/dimensión y panel de alertas.

## Configuraciones documentadas

- `config_demo.json`: configuración **final planificada** para la tesis (1 000 usuarios / 100 000 transacciones por semilla).
- `config_s8_demo.json`: configuración de la **evidencia S8** ya ejecutada (300 usuarios / 30 000 transacciones por semilla; seeds 42, 43 y 44).
- `run_quick_demo.bat`: demostración reducida de 120 usuarios para grabación o exposición en vivo.

## Ejecución rápida para la exposición

En Windows:

```bat
run_quick_demo.bat
```

La versión rápida usa 120 usuarios (12 000 transacciones) para reducir el tiempo de espera durante una demostración en vivo.

## Ejecución base del avance S8

```bat
run_demo.bat
```

o:

```bash
python run_demo.py --output outputs
```

Al terminar, abrir `outputs/reporte_demo.html` en el navegador.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En Linux/macOS, activar con `source .venv/bin/activate`.

## Estructura

- `src/simulator.py`: perfiles, transacciones, warm-up e inyección ATO.
- `src/features.py`: baseline congelado y 12 características.
- `src/models.py`: Isolation Forest, Autoencoder y LOF auxiliar.
- `src/evaluation.py`: umbral, métricas y estratificación.
- `src/report.py`: reporte HTML de demostración.
- `outputs/`: resultados de la corrida base.

## Importante

Las métricas generadas son **resultados reales de una corrida sintética inicial**, no resultados finales de la tesis. Sirven como evidencia de construcción S8. El experimento final deberá ejecutar el protocolo definitivo con múltiples semillas y análisis estadístico preespecificado.
