# 1. Resumen del Proyecto

La mayoría de los sistemas de recomendación modernos (Netflix, Amazon) se basan en Big Data y filtrado colaborativo. Este proyecto aborda el escenario opuesto: Small Data. Con un dataset de apenas unos cientos de registros, el desafío principal es extraer señal predictiva sin caer en el sobreajuste (overfitting).

## El core de la "Solución":

Para resolver este problema, el pipeline final se apoya en tres pilares arquitectónicos:

### Enriquecimiento de datos (OMDb API):
Se desarrolló un motor de ingesta que expande los datos básicos (título e ID) con metadatos técnicos: recaudación (BoxOffice), premios detallados, críticas de Rotten Tomatoes y Metascore, y votos en IMDb.

### Positional Multi-Label Encoder (Innovación Propia):
Las variables como "Actores" o "Directores" son problemáticas en datasets pequeños. El uso de One-Hot Encoding crearía miles de columnas vacías. En su lugar, diseñé un encoder que:
- Limpia y jerarquiza las etiquetas por frecuencia global.
- Calcula un Target Encoding suavizado para cada etiqueta.
- Genera un número fijo (K) de columnas basadas en la importancia, manteniendo la dimensionalidad bajo control y capturando la influencia de los nombres más relevantes.

### Adicion social:
Ante la constatación de que los metadatos objetivos no bastan para predecir el "gusto", se integraron valoraciones de otros usuarios (datos públicos cedidos voluntariamente). Esto permite al modelo encontrar patrones de afinidad subjetiva, actuando como un sustituto del filtrado colaborativo tradicional.

## Evaluación

Para este proyecto, el R2 tradicional no era suficiente. Se implementó una métrica personalizada que penaliza la inestabilidad:

Metric=μR2/0.5⋅(σR2⋅σMSE)⋅μMSE 

Esta fórmula asegura que el modelo elegido no solo sea preciso, sino robusto ante diferentes semillas de datos.


# 2. Requisitos, Ejecución y Artefactos

## Datos necesarios (inputs)

Para ejecutar el proyecto se requieren:

- `biblioteca-peliculas-20250905.csv` (dataset base)
- Datos de valoraciones de usuarios obtenidos mediante el script de scraping (carpeta `valoraciones/`)

> Nota: el scraping solo es necesario si se quiere regenerar las valoraciones. El repositorio incluye una snapshot de las valoraciones para reproducibilidad.

## Artefactos generados (outputs)

Al ejecutar el pipeline, se generan automáticamente archivos intermedios y resultados:

### Bibliotecas derivadas
- Biblioteca numérica
- Biblioteca con detalle
- Biblioteca limpia
- Biblioteca depurada

### Resultados de modelos
- Modelo lineal
- Random Forest
- AutoGluon

### Modelo final
- Modelo final entrenado

### Otros
- Archivos de soporte / intermedios no relevantes para el uso final del proyecto

Estos artefactos se generan durante la ejecución y no es necesario versionarlos manualmente.

# 3. Experimentos (algunos no incluidos)

Antes de llegar a la solución final, se exploraron múltiples vías de investigación. Aunque muchas fueron descartadas por introducir ruido o complejidad innecesaria, cada una aportó valor al entendimiento del dataset.

## Procesamiento de Lenguaje Natural (NLP) y Embeddings

Se intentó extraer el "sentimiento" y la temática de las películas a través de sus sinopsis (Plots):

- Sentence Transformers: Se utilizaron modelos como all-mpnet-base-v2 para convertir textos en vectores de 768 dimensiones.
- Reducción de Dimensionalidad: Se aplicó UMAP para proyectar estos vectores en 2D y 20D, buscando clusters de gustos. También se probó LDA (Latent Dirichlet Allocation) para extraer temas (ej: "drama familiar" vs "acción espacial").

Resultado: En un dataset pequeño, los embeddings tienden a capturar ruido semántico que no siempre correlaciona con la nota personal.

## Arquitecturas de Aprendizaje Automático Avanzado

- AutoML (AutoGluon & MLjar): Se ejecutaron pruebas con configuraciones de "Extreme Quality", incluyendo modelos experimentales como TabPFN. Estos sirvieron como benchmark para entender el techo predictivo del dataset.
- Mixture of Experts (MoE): Se construyó un ensamble donde diferentes "expertos" (Random Forests) se especializaban en sub-datasets (uno para actores, otro para directores, otro para métricas numéricas). La decisión final se tomaba mediante una media ponderada por el rendimiento Out-of-Fold.

## Learning to Rank (LTR) y Clasificación Pairwise

Se cambió el paradigma de regresión (predecir una nota) por uno de ranking (ordenar películas):

- Pairwise Labeling: Se creó una herramienta interactiva para ordenar películas manualmente y generar un dataset de comparaciones ("A es mejor que B").
- LGBM Ranker: Se entrenó un modelo de LightGBM para aprender estas jerarquías.
- Traductor de Rango: Se implementó una regresión lineal para convertir la posición en el ranking de vuelta a una nota de 1 a 10.

Resultado: Aunque el enfoque es potente, la escasez de datos de entrenamiento para el ranking limitó su generalización.

## Ingeniería de Características y Selección Rigurosa

- RFE con PFI: Se utilizó Eliminación Recursiva de Características combinada con Importancia por Permutación (Permutation Feature Importance). Esto permitió identificar qué variables realmente aportaban valor y cuáles eran puramente aleatorias.
- Interacciones Polinómicas: Se generaron combinaciones de segundo grado entre variables numéricas (ej: imdbRating * Metascore) para capturar efectos sinérgicos, filtrándolas posteriormente para evitar el overfitting.

# 4. Conclusiones y Futuro

El proyecto demuestra que, en entornos de Small Data, la ingeniería de características y la calidad de los datos (el "Data-Centric AI") superan en importancia a la complejidad del algoritmo. El modelo final es una herramienta funcional que sirve de guía para futuras visualizaciones.

## Próximos pasos:

- Transfer Learning: Explorar el entrenamiento previo en datasets de terceros con perfiles de gusto similares para transferir los pesos hacia el modelo personal, compensando así la falta de datos iniciales.
- Análisis individualizado: Implementar un sistema de segmentación por usuario similar al aplicado en las valoraciones propias, utilizando técnicas de Target Encoding para capturar sesgos específicos de cada perfil social.
- Refinamiento de variables: Investigar métodos avanzados de feature pruning para eliminar ruido redundante y mejorar la eficiencia computacional del modelo.
- Transformaciones No Lineales: Explorar funciones de transferencia como:

  f(x)=a⋅sinh(x−b)+c

  q(x)=kx+l

  w(x)=r⋅tanh(ix−t)+u

  s(x)=x^2

para ajustar la distribución de las predicciones y mejorar la sensibilidad en los extremos de la escala de notas. En particular, la transformación cuadrática busca corregir el sesgo de escala lineal, modelando la realidad perceptual donde la distancia crítica entre una nota de 9 y 10 es significativamente mayor (y más relevante para la recomendación) que la diferencia existente entre un 1 y un 2.
