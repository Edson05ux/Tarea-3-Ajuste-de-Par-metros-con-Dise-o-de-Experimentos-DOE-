# Tarea 3 – Ajuste de temperatura con DOE

Proyecto independiente de la Tarea 2. Implementa la Opción A: diseño factorial completo con tres niveles de temperatura inicial T₀, tres tamaños de TSP (n = 5, 10, 15) y diez réplicas por combinación (90 corridas).

## Ejecución

1. Instala las dependencias con `python -m pip install -r requirements.txt`.
2. Ejecuta `python src/Ejercicios/Ejercicio_DOE_temperatura.py` para correr el experimento.

El experimento genera el ANOVA, el CSV con las 90 observaciones, el resumen de óptimos, el bootstrap por punto de diseño y las gráficas en `src/Ejercicios/graficas/`. El reporte final no se genera automáticamente; prepara el entregable según las indicaciones del curso.

## Reproducibilidad

La semilla inicial está configurada con la matrícula del estudiante (`EXPERIMENT_SEED = 2151976`) y determina el único valor aleatorio r del diseño. Este r se genera una sola vez y se reutiliza en todos los niveles y corridas.
