import pandas as pd
import matplotlib.pyplot as plt
import os

# Directorio que contiene los archivos CSV
csv_directory = "D:/programa/Pareto"

# Obtener la lista de archivos CSV en el directorio
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

# Crear un DataFrame combinando los datos de todos los archivos CSV en el directorio
# Especificar el separador de columnas como ';'
df_pareto = pd.concat([pd.read_csv(os.path.join(csv_directory, file), sep=';') for file in csv_files], ignore_index=True)

# Guardar las primeras 100 líneas del DataFrame combinado en un archivo CSV llamado Pareto.csv
# Especificar el separador de decimales en la columna Percent como '.'
df_pareto.head(100).to_csv("Pareto.csv", index=False, sep=';', decimal='.')

# Proceso para obtener ResumenPareto.csv
# Guardar las primeras 100 líneas del DataFrame combinado en un archivo CSV llamado ResumenPareto.csv
# Especificar el separador de decimales en la columna Percent como '.'
df_pareto.head(100).to_csv("ResumenPareto.csv", index=False, sep=';', decimal='.')

# Calcular el Pareto utilizando ResumenPareto.csv
# Supongamos que tienes una columna llamada "cantidad" que representa la frecuencia de cada categoría.
# Ajusta la columna según tus datos.
df_pareto_sorted = df_pareto.head(100).sort_values(by='Percent', ascending=False)
df_pareto_sorted['cumulative_percentage'] = df_pareto_sorted['Percent'].cumsum() / df_pareto_sorted['Percent'].sum() * 100

# Graficar el resultado de manera elegante
fig, ax1 = plt.subplots(figsize=(10, 6))

# Barra de frecuencia
color = 'tab:blue'
ax1.bar(df_pareto_sorted['Emotions'], df_pareto_sorted['Percent'], color=color, alpha=0.7, label='Frecuencia')
ax1.set_xlabel('Emotions')
ax1.set_ylabel('Percent', color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Línea de porcentaje acumulado
ax2 = ax1.twinx()
color = 'tab:red'
ax2.plot(df_pareto_sorted['Emotions'], df_pareto_sorted['cumulative_percentage'], color=color, marker='o', label='Porcentaje Acumulado')
ax2.set_ylabel('Porcentaje Acumulado', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Configuraciones generales
plt.title('Pareto Chart')
fig.tight_layout()
plt.xticks(rotation=45, ha='right')
plt.show()

