# el siguient programa al recibir gran cantidad de datos, y graficarlos con el metodo de pareto, el grafico no es muy legible.
# Puedes ubicar un procedimiento en el codigo que permita hacer legible el grafico?
# El codigo es:

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_pareto_curve(directory_path):
    # Obtener la lista de archivos en el directorio
    files = [file for file in os.listdir(directory_path) if file.endswith(".csv")]

    # Iterar sobre cada archivo en el directorio
    for file in files:
        file_path = os.path.join(directory_path, file)

        # Leer el archivo CSV con Pandas
        df = pd.read_csv(file_path)

        # Ordenar el DataFrame por la columna 'Percent' de forma descendente
        df.sort_values(by='Percent', ascending=False, inplace=True)

        # Calcular el porcentaje acumulado
        df['Cumulative Percent'] = df['Percent'].cumsum() / df['Percent'].sum() * 100

        # Graficar la curva de Pareto
        plt.figure(figsize=(14, 8))  # Aumentar el tamaño de la figura para mejorar la legibilidad
        plt.bar(df['Emotions'], df['Percent'], color='blue')
        plt.plot(df['Emotions'], df['Cumulative Percent'], color='red', marker='o')
        plt.xlabel('Emotions')
        plt.ylabel('Percent')
        plt.title(f'Pareto Chart - {file}')
        plt.xticks(rotation=45, ha='right', fontsize=10)  # Ajustar el tamaño y ángulo de las etiquetas
        plt.yticks(fontsize=10)  # Ajustar el tamaño de las etiquetas del eje y
        plt.legend(['Cumulative Percent'], loc='upper left', fontsize=10)  # Ajustar el tamaño de la leyenda

        # Mostrar la gráfica
        plt.tight_layout()  # Mejorar el espaciado entre elementos de la figura
        plt.show()

# Directorio que contiene los archivos CSV
directory_path = "C:/videos/experimentos 10-01-2024/Pareto/"
plot_pareto_curve(directory_path)
