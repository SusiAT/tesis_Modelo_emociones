# tesis_Modelo_emociones
Describe todos los experimentos realizados en la tesis Modelo ubicuo de análisis emocional en clases en línea
# Descripción del Código
El código es un script en Python que utiliza diversas bibliotecas como Keras, OpenCV, Owlready2, Matplotlib, y NetworkX para analizar expresiones faciales en videos, calcular porcentajes de emociones, realizar K-Means clustering, y construir una ontología de Action Tendency con Emociones.
Carga de Bibliotecas:

Importa las bibliotecas necesarias como Keras, OpenCV, Owlready2, Matplotlib, NetworkX, entre otras.
Definición de Rutas y Parámetros:

Define rutas para el modelo de detección de rostros, el modelo de emociones, y el directorio de videos.
Establece parámetros para la detección de rostros y emociones.
Procesamiento de Videos:

Lee y procesa videos desde un directorio.
Detecta rostros y predice emociones utilizando un modelo preentrenado.
Calcula porcentajes de emociones y los guarda en archivos CSV.
Análisis de Datos:

Utiliza pandas para organizar y procesar datos.
Aplica K-Means clustering para agrupar emociones.
Genera un informe con resultados y estadísticas.
Construcción de Ontología:

Utiliza Owlready2 para construir una ontología de Action Tendency con Emociones.
Asocia clases de Action Tendency con emociones específicas.
Creación de Gráfico de Red:

Utiliza NetworkX y Matplotlib para crear un gráfico de red representando la ontología.
Guarda el gráfico como una imagen.
Resultados:

Guarda resultados y estadísticas en archivos de texto y CSV.
Observaciones:

Se observa la conexión entre las emociones y las clases de Action Tendency según la teoría de Frijda.
