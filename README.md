# tesis_Modelo_emociones
Describe todos los experimentos realizados en la tesis Modelo ubicuo de análisis emocional para clases en línea reunidos en un solo programa.
# Descripción del Código
El código es un script en Python que utiliza diversas bibliotecas como Keras, OpenCV, Owlready2, Matplotlib, y NetworkX para identificar las emociones básicas en el rostro de los estudiantes, determinar emociones predominantes, verificar si las emociones básicas pueden crear clusters emocionales y finalmente usar estos clusters para inferir tendencias a la acción para el aprendizaje mediante una interpretación semántica ontológica durante clases en línea grabadas en video, que sirvan de base para la retroalimentación en el proceso de enseñanza – aprendizaje.

Carga de Bibliotecas:

    Importa las bibliotecas necesarias como Keras, OpenCV, Owlready2, Matplotlib, NetworkX, entre otras.
    El programa requiere Python 3.11 instalado en tu equipo.
       - Descarga Python desde https://www.python.org/.
       - Durante la instalación, asegúrate de habilitar la opción "Add Python to PATH".
       - Verifica la instalación abriendo una terminal o consola y escribiendo: python --version.
    
Definición de Rutas y Parámetros:

    Instala las bibliotecas necesarias mediante pip. Ejecuta el siguiente comando en la terminal:
    pip install tkinter pandas matplotlib keras imutils opencv-python owlready2 scikit-learn Jena java lib and others: https://jena.apache.org/
    Define rutas para el modelo de detección de rostros, el modelo de emociones, y el directorio de videos.
       
Procesamiento de Videos:

    Lee y procesa videos desde un directorio.
    Detecta rostros y predice emociones utilizando un modelo preentrenado.
    Calcula porcentajes de emociones y los guarda en archivos csv.
    Trabajamos con el Software base Emotion Recognition utilizado para registrar el rostro humano y las emociones encontradas y obtener las 
    probabilidades de estas emociones. (https://github.com/omar-aymen/Emotion-recognition#p1)
    
![image](https://github.com/user-attachments/assets/1073b38e-ecb1-4267-a08a-3615cabdae86)
    
Análisis de Datos:

    Utiliza pandas para organizar y procesar datos.
    Aplica KMEANS clustering para agrupar emociones.
    
Construcción de Ontología:

    Utiliza Owlready2 para construir una ontología de Action_Tendency con Emociones.
    Asocia clases de Action_Tendency con emociones básicas.
    
Creación de Gráfico de Red:

    Utiliza NetworkX y Matplotlib para crear un gráfico de red representando la ontología.
    Guarda el gráfico como una imagen.
    
Resultados:

    Guarda resultados y estadísticas en archivos de texto y CSV.

![image](https://github.com/user-attachments/assets/92eb22bb-a908-49b8-86d2-7f08b36889a6)

La metodología del modelo ubicuo de análisis emocional, aplicada en clases en línea, permite registrar y analizar emociones detectadas en los rostros de los estudiantes, así como también inferir y validar las tendencias a la acción para el aprendizaje.

El programa permite ejecutar dos procesos principales:
   - Action-Tendency:  realiza el análisis de cada video empezando por la identificación de las emociones básicas mediante el software 
    Emotion Recognition, para luego crear clusters de emociones que servirán para instanciar la ontología HASIO hasta obtener finalmente 
    las tendencias a la acción según la teoría de Frijda.
   - Pareto: realiza el análisis de las emociones predominantes de todos los videos grabados; es decir identifica las emociones predominantes
    de todo el grupo de estudiantes y finalmente lo grafica mediante un diagrama de Pareto.
  
Archivos del Programa

      El programa requiere los siguientes archivos en sus rutas correspondientes:
      
      - mubicuo.py: Es el frontal principal que llama a los programas Action-Tendency y Pareto.
      - autoridad.py: Para el análisis de Action-Tendency.
        - Ubicación esperada: D:/programa/autoridad.py
      - Pareto.py: Para generar gráfico de Pareto.
        - Ubicación esperada: D:/programa/Pareto/Pareto.py
      - HASIO.owl: Archivo de ontología utilizado en Action-Tendency.
        - Ubicación esperada: D:/programa/HASIO.owl
      - Videos de los estudiantes: Ubicados en D:/programa/videosestudiantes/
  
  Ruta de Resultados
  
      Los resultados generados se guardan en las siguientes carpetas, dependiendo del proceso ejecutado:
      - Action-Tendency: En D:/programa/resultados/[nombre_archivo]/
      - Pareto: En D:/programa/resultados/Pareto/
    
Instrucciones de Ejecución

      1. Asegúrate de tener el archivo principal con el código (mubicuo.py) y colócalo en una carpeta accesible.
      2. Abre una terminal o consola.
      3. Navega a la carpeta donde está el archivo mubicuo.py.
      4. Ejecuta el programa con python mubicuo.py.

A continuación, se presentan las imágenes que demuestran los pasos antes mencionados.

1. Interfaz principal

![image](https://github.com/user-attachments/assets/818354e9-0301-4c56-8191-b43cff69dbe2)

2. Al elegir la opción _Action_Tendency_ se identifica las emociones básicas usando el software de reconocimiento de emociones para cada estudiante, para luego agrupar estas emociones aplicando el algoritmo KMEANS. Finalmente, estos clusters emocionales se instancian en la ontología HASIO y se obtiene las tendencias a la acción por cada estudiante

![image](https://github.com/user-attachments/assets/24cf2732-de76-498b-99ca-a9465c5348ae)

![image](https://github.com/user-attachments/assets/79e431b5-29df-459c-8ca5-002251dabb82)

![image](https://github.com/user-attachments/assets/35acc9be-453e-48df-943e-60a874e94d60)

3. Al elegir la opción _Pareto_ se identifica las emociones predominantes de todos los estudiantes del curso mediante el diagrama de Pareto.
    
![image](https://github.com/user-attachments/assets/eefd2bbe-929a-4c24-a00b-cb277ae87800)

Contacto para Soporte
        Nombre: Susana Arias
        Correo Electrónico: susi.alexa@gmail.com
        
