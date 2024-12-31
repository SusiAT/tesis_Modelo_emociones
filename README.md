# tesis_Modelo_emociones
Describe todos los experimentos realizados en la tesis Modelo ubicuo de análisis emocional para clases en línea reunidos en un solo programa.
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

1. El programa permite ejecutar dos procesos principales:
  - Action-Tendency: Análisis detallado del comportamiento de los estudiantes mediante videos, algoritmos de clustering y tendencias a la acción.
  - Pareto: Generación del gráfico de Pareto para analizar las emociones predominantes de todos los estudiantes.
  
2. Requisitos Previos
   
  2.1. Instalación de Python
  El programa requiere Python 3.11 instalado en tu equipo.
  1. Descarga Python desde https://www.python.org/.
  2. Durante la instalación, asegúrate de habilitar la opción "Add Python to PATH".
  3. Verifica la instalación abriendo una terminal o consola y escribiendo: python --version.
   
  2.2. Instalación de Bibliotecas Requeridas
  Instala las bibliotecas necesarias mediante pip. Ejecuta el siguiente comando en la terminal:
  pip install tkinter pandas matplotlib keras imutils opencv-python owlready2 scikit-learn Jena java lib and others: https://jena.apache.org/

  2.3. Archivos del Programa
  El programa requiere los siguientes archivos en sus rutas correspondientes:
  - autoridad.py: Para el análisis de Action-Tendency.
    - Ubicación esperada: D:/programa/autoridad.py
  - Pareto.py: Para generar gráficos de Pareto.
    - Ubicación esperada: D:/programa/Pareto/Pareto.py
  - HASIO.owl: Archivo de ontología utilizado en Action-Tendency.
    - Ubicación esperada: D:/programa/HASIO.owl
  - Videos de los estudiantes: Ubicados en D:/programa/videosestudiantes/
  
  2.4. Ruta de Resultados
  Los resultados generados se guardan en las siguientes carpetas, dependiendo del proceso ejecutado:
  - Action-Tendency: En D:/programa/resultados/[nombre_archivo]/
  - Pareto: En D:/programa/resultados/Pareto/
    
3. Instrucciones de Ejecución
  1. Asegúrate de tener el archivo principal con el código (mubicuo.py) y colócalo en una carpeta accesible.
  2. Abre una terminal o consola.
  3. Navega a la carpeta donde está el archivo mubicuo.py.
  4. Ejecuta el programa con python mubicuo.py.

![image](https://github.com/user-attachments/assets/818354e9-0301-4c56-8191-b43cff69dbe2)

La opción Action_Tendency realiza el análisis de cada video empezando por la identificación de las emociones básicas mediante el software Emotion Recognition, para luego crear clusters de emociones que servirán para instanciar la ontología HASIO hasta obtener finalmente las tendencias a la acción.
A continuación, se presentan las imágenes que demuestran los pasos antes mencionados.

![image](https://github.com/user-attachments/assets/24cf2732-de76-498b-99ca-a9465c5348ae)

![image](https://github.com/user-attachments/assets/79e431b5-29df-459c-8ca5-002251dabb82)

![image](https://github.com/user-attachments/assets/35acc9be-453e-48df-943e-60a874e94d60)

Mientras que la segunda opción Pareto realiza el análisis de las emociones predominantes de todos los videos grabados; es decir identifica las emociones predominantes de todo el grupo de estudiantes y finalmente lo grafica mediante un diagrama de Pareto.
A continuación, se presenta la captura de pantalla.

![image](https://github.com/user-attachments/assets/eefd2bbe-929a-4c24-a00b-cb277ae87800)

4. Posibles Errores y Soluciones
Consulta la sección de errores en el manual proporcionado.
5. Contacto para Soporte
Nombre: Susana Arias
Correo Electrónico: susi.alexa@gmail.com
6. Diagrama de flujo
Diagrama de flujo para la instalación del programa de análisis emocional para clases en línea.

![image](https://github.com/user-attachments/assets/6510781e-ea48-47dc-badf-4c27dc7510e6)


![image](https://github.com/user-attachments/assets/26867f32-a052-4a8d-96b8-ecb575da57a2)




