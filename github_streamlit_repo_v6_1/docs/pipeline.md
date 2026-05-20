# Pipeline

## Data input
Radiografías frontales IU X-Ray + indication clínica + UID.

## Preprocessing
Filtrado frontal, normalización, auditoría de etiquetas, split por UID, exclusiones.

## Model
Modelo visual calibrado como principal; texto/fusión como exploratorio.

## Training
Train/validation/calibration/threshold/test + CV por UID.

## Evaluation
ROC-AUC, PR-AUC, Brier, sensibilidad, especificidad, FPR, métricas adicionales, bootstrap, McNemar.

## Final output
Modelo principal, tablas, gráficos, demo y GitHub package.
