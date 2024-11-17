# Importar bibliotecas necesarias
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Cargar el archivo CSV
file_path = ("C:/videos/experimentos 10-01-2024/estudiante 1/e1roc.csv")  # Reemplaza "tu_archivo.csv" con la ruta real de tu archivo
df = pd.read_csv(file_path,sep=";")

# Definir las etiquetas reales (y_true) y las probabilidades predichas (y_scores)
y_true = df['Experto']
y_scores = df['Emociones']

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Calcular el área bajo la curva (AUC)
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()


