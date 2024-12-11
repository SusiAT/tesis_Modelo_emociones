# tesis_Modelo_emociones
Describe todos los experimentos realizados en la tesis Modelo ubicuo de análisis emocional para clases en línea
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
![image](https://github.com/user-attachments/assets/92eb22bb-a908-49b8-86d2-7f08b36889a6)

Trabajamos con: Software base utilizado para registrar el rostro humano y las emociones encontradas y obtener las probabilidades de estas emociones. (https://github.com/omar-aymen/Emotion-recognition#p1).

![image](https://github.com/user-attachments/assets/1073b38e-ecb1-4267-a08a-3615cabdae86)

Requerimientos

Product Version: Apache NetBeans IDE 11.3
Java: 1.8.0_333; Java HotSpot(TM) 64-Bit Server VM 25.333-b02
System: Windows 11 version 10.0 running on amd64; Cp1252; es_EC (nb)
Jena java lib and others: https://jena.apache.org/

SCREENSHOTS

![image](https://github.com/user-attachments/assets/95311814-ba01-49f6-9455-6b3d99c224da)
![image](https://github.com/user-attachments/assets/36fbd765-517d-4eb2-83ca-a53b5a6f857b)



