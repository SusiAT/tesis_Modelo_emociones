import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Directorio que contiene los archivos CSV
csv_directory = "C:/programa/codo"

# Obtener la lista de archivos CSV en el directorio
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

# Inicializar una lista para almacenar los valores de codo
all_distortions = []

# Iterar sobre cada archivo CSV en el directorio
for file in csv_files:
    # Crear un DataFrame para el archivo CSV actual
    df = pd.read_csv(os.path.join(csv_directory, file), sep=';', decimal='.')

    # Asignar una codificación numérica a cada emoción
    emocion_numeros = {emocion: numero for numero, emocion in enumerate(df['Emotions'].unique())}
    df['emocion_numero'] = df['Emotions'].map(emocion_numeros)

    # Calcular el método del codo para determinar el número óptimo de clusters (k) en KMeans
    distortions = []
    K_range = range(1, 11)  # Probamos con un rango de 1 a 10 clusters
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        # Utilizamos las dos columnas para el clustering
        kmeans.fit(df[['Percent', 'emocion_numero']])
        distortions.append(kmeans.inertia_)

    # Graficar y guardar el resultado del método del codo
    plt.plot(K_range, distortions, 'bx-')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Distorsión')
    plt.title(f'Método del Codo para KMeans - {file}')
    plt.savefig(f'{os.path.splitext(file)[0]}_codo.jpg')
    plt.show()

    # Almacenar los valores de codo para calcular el promedio
    all_distortions.extend(distortions)

# Calcular y presentar el promedio de todos los valores de codo
average_distortion = sum(all_distortions) / len(all_distortions)
print(f"Promedio de los valores de codo: {average_distortion}")
