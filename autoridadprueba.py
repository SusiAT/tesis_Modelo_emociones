from keras.preprocessing.image import img_to_array
from pathlib import Path
import imutils
import cv2
from keras.models import load_model
import numpy as np
import time
import os
import time
import locale
from owlready2 import *
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.cluster import KMeans




locale.setlocale(locale.LC_NUMERIC, 'es_ES.UTF-8')

# parameters for loading data and images
home = os.path.dirname(__file__)
detection_model_path = home+'/'+'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = home+'/'+'models/_mini_XCEPTION.102-0.66.hdf5'
# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]


#feelings_faces = []
#for index, emotion in enumerate(EMOTIONS):
   # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))


#file

# Especifica la ruta del directorio que deseas listar
ruta_directorio = "C:/programa/videosestudiantes"
   

def mostrar_nombres_de_archivos(ruta_directorio):
    # Obtener la lista de nombres de archivos en el directorio
    nombres_archivos = os.listdir(ruta_directorio)
    print(nombres_archivos)

    # Mostrar cada nombre de archivo en pantalla
    for nombre_archivo in nombres_archivos:
        print(nombre_archivo)
        fName= nombre_archivo+".csv"
        rutaalnuevo="C:/programa/resultados/"+nombre_archivo
        #os.mkdir(rutaalnuevo)
        #f = open(rutaalnuevo+"/" + fName, 'w')
        #titulo = "{};{}".format("Emotions", "Percent")
        #f.write(titulo)
        #f.close
        #print(fName+'@@'+home + '/' + fName)

        # starting video streaming
        #cv2.namedWindow('face')
        #camera = cv2.VideoCapture(0)
        #camera = cv2.VideoCapture('C:/programa/videosestudiantes/'+nombre_archivo)
        #cont = 0
        print("pase")
    

       

        # Leer el archivo CSV
        df = pd.read_csv(rutaalnuevo+"/" + fName, sep=';', usecols=['Percent'])
        print("HOLA")
        print(df)

        # Procesar los valores de Percent y convertirlos a numéricos
        df['Percent'] = df['Percent'].apply(lambda x: float(str(x)[:5]))

        # Inicializar una lista para almacenar las filas del DataFrame de salida
        output_data = []

        # Iterar sobre los valores de Percent y organizarlos en columnas
        for i in range(0, len(df), 7):
            output_data.append(df['Percent'].iloc[i:i+7].values)

        # Crear un DataFrame a partir de los datos procesados
        processed_df = pd.DataFrame(output_data, columns=['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral'])

        # Guardar el DataFrame en un archivo CSV con separador ";"
        processed_df.to_csv(rutaalnuevo+"/matriz.csv", sep=';', index=False)




        # Nombre del archivo CSV de entrada y salida
        archivo_entrada = rutaalnuevo+"/matriz.csv"
        archivo_salida = rutaalnuevo+"/matrizk.csv"

        # Leer el archivo CSV usando punto y coma como separador
        df = pd.read_csv(archivo_entrada, sep=';')

        # Convertir los valores a números
        df = df.apply(pd.to_numeric, errors='coerce')

        # Guardar el DataFrame resultante en un nuevo archivo CSV
        df.to_csv(archivo_salida, sep=';', index=False)

        print("Se ha guardado el archivo CSV con los valores convertidos a números:", archivo_salida)



      

        #:::::::::::::::::::::::::::::::::::::::::::

        # Leer el archivo CSV
        df = pd.read_csv(rutaalnuevo+"/matrizk.csv", sep=';', decimal='.')

        # Obtener las emociones de la primera fila
        emociones = df.columns

        # Eliminar la primera fila del DataFrame
        df = df.drop(0)

        # Calcular K-Means con k=4
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(df)

        # Obtener los centroides finales
        centroides = kmeans.cluster_centers_


        # Obtener las etiquetas de clúster asignadas a cada instancia
        etiquetas_clusters = kmeans.labels_

        # Calcular la emoción más representativa para cada clúster
        emociones_cluster = []
        valores_cluster = []
        instancias_cluster = []


        for i, centroide in enumerate(centroides):
            indice_emocion = np.argmax(centroide)
            emocion = emociones[indice_emocion]
            valor = centroide[indice_emocion]
            emociones_cluster.append(emocion)
            valores_cluster.append(valor)
    
            # Contar instancias en el clúster
            instancias_en_cluster = np.sum(etiquetas_clusters == i)
            instancias_cluster.append(instancias_en_cluster)

        # Clasificar emociones en MAYORES o MENORES según el número de instancias por cluster
        clasificacion_emociones = ["MAYOR" if instancias >= np.median(instancias_cluster) else "MENOR" for instancias in instancias_cluster]

        # Mostrar la matrizk
        print("Matrizk:")
        print(df)

        # Mostrar los resultados
        print("\nResultados:")
        f = open(rutaalnuevo+"/resultadosKmeans.txt", 'w')
        for i, (emocion, valor, instancias, clasificacion) in enumerate(zip(emociones_cluster, valores_cluster, instancias_cluster, clasificacion_emociones)):
            print(f"Cluster {i + 1}: Emoción: {emocion}, Valor: {valor}, Instancias: {instancias}, Clasificación: {clasificacion}, Centroides:{centroides[i]}")
            text2="Cluster: " + str(i+1)+ " Emoción: "+ emocion + " Valor: " + str(valor) + " Instancias:" + str(instancias)+ " Clasificación:"+ clasificacion + " Centroides:" + str(centroides[i])
            print(text2) 
            f.write(text2+"\n")
        f.close()

        

        # Cargar la ontología desde el archivo local
        onto = get_ontology("FERO.owl").load()

        # Definir clases y propiedades
        with onto:
            class Mental_State(Thing):
                pass
    
            class Affective_State(Mental_State):
                pass
    
            class Emotion(Affective_State):
                pass
    
            class Action_Tendency(Thing):
                pass
    
            class Frijda_Action_Tendency(Action_Tendency):
               pass
    
            class Agonistic(Frijda_Action_Tendency):
                pass
    
            class Approach(Frijda_Action_Tendency):
                pass
    
            class Avoidance(Frijda_Action_Tendency):
                pass
    
            class BeingWith(Frijda_Action_Tendency):
                pass
    
            class Dominating(Frijda_Action_Tendency):
                pass
    
            class Interrupting(Frijda_Action_Tendency):
                pass
    
            class NoAttending(Frijda_Action_Tendency):
                pass
    
            class Rejecting(Frijda_Action_Tendency):
                pass
    
            class Submitting(Frijda_Action_Tendency):
                pass

        # Diccionario que mapea las clases de acción tendencial con las emociones correspondientes
        emotion_action_mapping = {
            Agonistic: ["anger"],
            Approach: ["joy"],
            Avoidance: ["fear"],
            BeingWith: ["happy"],
            Dominating: ["anger", "disgust"],
            Interrupting: ["surprise"],
            NoAttending: ["sad"],
            NoAttending: ["neutral"],
            Rejecting: ["disgust", "anger"],
            Submitting: ["sad", "fear"]
        }


        

        # Cargar la ontología desde el archivo local
        onto = get_ontology("FERO.owl").load()

        # Definir clases y propiedades
        with onto:
            class Mental_State(Thing):
                pass
    
            class Affective_State(Mental_State):
                pass
    
            class Emotion(Affective_State):
                pass
    
            class Action_Tendency(Thing):
                pass
    
            class Frijda_Action_Tendency(Action_Tendency):
                pass
    
            class Agonistic(Frijda_Action_Tendency):
                pass
    
            class Approach(Frijda_Action_Tendency):
                pass
    
            class Avoidance(Frijda_Action_Tendency):
                pass
    
            class BeingWith(Frijda_Action_Tendency):
                pass
    
            class Dominating(Frijda_Action_Tendency):
                pass
    
            class Interrupting(Frijda_Action_Tendency):
                pass
    
            class NoAttending(Frijda_Action_Tendency):
                pass
    
            class Rejecting(Frijda_Action_Tendency):
                pass
    
            class Submitting(Frijda_Action_Tendency):
                pass

        # Mapeo de emociones a action tendencies
        emotion_to_action_tendency = {
            "angry": Agonistic,
            "joy": Approach,
            "scared": Avoidance,
            "happy": BeingWith,
            "surprised": Interrupting,
            "sad": NoAttending,
            "neutral":NoAttending,
            "disgust": Rejecting,
            "happy": Approach,  # Se repite para happy ya que en el mapa se asocia tanto con Approach como con NoAttending
        }

        # Iterar sobre los centroides y crear instancias de Emotion
        for emocion2, valor in zip(emociones_cluster, valores_cluster):
            # Obtener la clase de Action Tendency correspondiente a la emoción
            action_tendency_class = emotion_to_action_tendency.get(emocion2.lower())
    
            # Si se encuentra la clase de Action Tendency, crear la instancia correspondiente y establecer la relación isActionTendencyOf
            if action_tendency_class:
                # Crear una instancia de la clase Emotion
                emocion_instance = Emotion(emocion2)
                # Asignar el valor deseado al individuo
                emocion_instance.value = emocion2
        
                # Crear una instancia de la clase de Action Tendency y establecer la relación isActionTendencyOf
                action_tendency_instance = action_tendency_class()
                action_tendency_instance.isActionTendencyOf.append(emocion_instance)
        
                print(f"Se ha creado una instancia de {action_tendency_class.__name__} y se ha establecido la relación isActionTendencyOf con la instancia {emocion2}.")

        # Guardar la ontología modificada
        onto.save(rutaalnuevo+"/FERO_modified.owl")



      
        

        # Cargar la ontología desde el archivo local
        onto = get_ontology(rutaalnuevo+"/FERO_modified.owl").load()

        # Definir clases y propiedades
        with onto:
            class Mental_State(Thing):
                pass
    
            class Affective_State(Mental_State):
                pass
    
            class Emotion(Affective_State):
                pass
    
            class Action_Tendency(Thing):
                pass
    
            class Frijda_Action_Tendency(Action_Tendency):
                pass
    
            class Agonistic(Frijda_Action_Tendency):
                pass
    
            class Approach(Frijda_Action_Tendency):
               pass
    
            class Avoidance(Frijda_Action_Tendency):
                pass
    
            class BeingWith(Frijda_Action_Tendency):
                pass
    
            class Dominating(Frijda_Action_Tendency):
                pass
    
            class Interrupting(Frijda_Action_Tendency):
                pass
    
            class NoAttending(Frijda_Action_Tendency):
                pass
    
            class Rejecting(Frijda_Action_Tendency):
                pass
    
            class Submitting(Frijda_Action_Tendency):
                pass

        # Crear un grafo dirigido
        G = nx.DiGraph()

        # Obtener todas las clases de Action Tendency
        action_tendency_classes = [Agonistic, Approach, Avoidance, BeingWith, Dominating, Interrupting, NoAttending, Rejecting, Submitting]

        # Agregar las clases madre de Action Tendency
        G.add_node("Action Tendency")

        # Diccionario que mapea las clases de Action Tendency con las emociones correspondientes
        emotion_mapping = {
            Agonistic: "Anger",
            Approach: "Joy",
            Avoidance: "Fear",
            BeingWith: "Happy",
            Dominating: "Anger, Disgust",
            Interrupting: "Surprise",
            NoAttending: "Sad",
            NoAttending: "Neutral",
            Rejecting: "Disgust, Anger",
           Submitting: "Sad, Fear"
        }

        # Iterar sobre las clases de Action Tendency
        for action_tendency_class in action_tendency_classes:
            # Verificar si la clase tiene instancias creadas
            if len(action_tendency_class.instances()):
                # Obtener la emoción asociada a esta clase
                emotion = emotion_mapping.get(action_tendency_class, "Unknown")
        
                # Agregar el nombre de la clase al grafo
                G.add_node(action_tendency_class.__name__, label=emotion, shape='rectangle')
                G.add_edge("Action Tendency", action_tendency_class.__name__)
        
                # Obtener las instancias de esta clase
                instances = action_tendency_class.instances()
        
                # Iterar sobre las instancias y agregarlas al grafo
                for instance in instances:
                    # Obtener la emoción básica asociada a esta instancia
                    basic_emotion = instance.isActionTendencyOf[0].value
                    # Agregar la instancia al grafo con la etiqueta de la emoción básica
                    G.add_node(basic_emotion, shape='ellipse')
                    G.add_edge(action_tendency_class.__name__, basic_emotion)

        # Dibujar el grafo
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=42)  # Layout para el grafo
        nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, font_weight="bold", arrowsize=20, node_color='skyblue', edge_color='gray', width=2, alpha=0.7, connectionstyle='arc3, rad = 0.1')
        plt.title("Ontología de Action Tendency con Emociones")
        plt.savefig(rutaalnuevo+"/FERO_action_tendency_graph.jpg")  # Guardar como JPG
        plt.show()




        


mostrar_nombres_de_archivos(ruta_directorio)