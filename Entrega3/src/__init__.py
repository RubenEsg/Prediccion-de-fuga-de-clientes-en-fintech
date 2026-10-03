"""Pipeline reproducible del proyecto de fuga de clientes (Entregable 3).

Módulos
-------
config
    Semilla global, constantes del diseño experimental y localización de datos.
datos
    Carga, auditoría, variables derivadas, partición y construcción de la etiqueta.
preprocesamiento
    ColumnTransformer (escalado + one-hot) que se ajusta solo con entrenamiento.
modelos
    Catálogo de los 7 modelos de clasificación y 7 de regresión con sus espacios de búsqueda.
balanceo
    Técnicas de balanceo y ensamblado del pipeline preprocesado → muestreo → modelo.
optimizacion
    Grid, Random, Bayesiana (Optuna) y Genética (DEAP) con validación cruzada anidada.
evaluacion
    Métricas de clasificación y regresión.
registro
    Tabla maestra de experimentos persistida en disco.
experimento
    Orquestación de una corrida (tarea, modelo, balanceo, optimizador) y del producto completo.
"""
