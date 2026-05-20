# Guion de demo en vivo

1. Explica que el entrenamiento ya fue ejecutado y que la demo muestra inferencia/evaluación retrospectiva.
2. Muestra un caso true positive: imagen, indication, probabilidad, umbral y decisión.
3. Muestra un caso false positive para discutir limitaciones.
4. Recalca que el modelo principal es `image_calibrated` y el umbral operativo es `min_fpr_recall_0.90`.
5. Frase sugerida: "La demo no reentrena; ejecuta la etapa de inferencia/evaluación sobre casos reservados de test".
