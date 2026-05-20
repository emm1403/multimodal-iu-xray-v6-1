# Resumen ejecutivo V6.1

## Resultado principal

Modelo recomendado: `image_calibrated`  
Umbral operativo: `min_fpr_recall_0.90`

| Métrica UID | Valor |
|---|---:|
| ROC-AUC | 0.857 |
| PR-AUC | 0.744 |
| Brier | 0.132 |
| Sensibilidad | 92.5% |
| Especificidad | 51.2% |
| FPR | 48.8% |
| F1 | 0.593 |
| TN / FP / FN / TP | 84 / 80 / 5 / 62 |

## Pipeline

- **Data input:** radiografía frontal IU X-Ray + `indication` clínica.
- **Preprocessing:** normalización, split por UID, auditoría y exclusión de casos ambiguos/incidental/resueltos.
- **Model:** rama visual calibrada; texto y fusión como análisis exploratorio.
- **Training:** splits separados y CV por UID.
- **Evaluation:** ROC-AUC, PR-AUC, Brier, sensibilidad, especificidad, FPR, métricas adicionales, bootstrap y McNemar.
- **Final output:** assets de presentación, repo GitHub-ready y demo retrospectiva.

## Multimodalidad

Peso seleccionado para fusión ponderada: `image_weight=1.0, text_weight=0.0`. La señal textual basada solo en `indication` no aportó mejora robusta frente a imagen calibrada.
