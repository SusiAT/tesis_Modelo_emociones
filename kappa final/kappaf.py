import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score

def calcular_kappa_desde_csv(ruta_experto1, ruta_experto2):
    # Input validation
    try:
        experto1 = pd.read_csv(ruta_experto1, sep=";")
        experto2 = pd.read_csv(ruta_experto2, sep=";")
    except FileNotFoundError:
        print("Error: One or both CSV files not found.")
        return None, None

    columnas_comunes = set(experto1.columns).intersection(experto2.columns)

    resultados_kappa = {}

    for columna in ['Mayores1', 'Mayores2', 'Menores1', 'Menores2']:
        if columna in columnas_comunes:
            respuestas_experto1 = experto1[columna].tolist()
            respuestas_experto2 = experto2[columna].tolist()

            matriz_confusion = confusion_matrix(respuestas_experto1, respuestas_experto2)
            kappa = cohen_kappa_score(respuestas_experto1, respuestas_experto2)

            resultados_kappa[columna] = {
                'Matriz de Confusión': matriz_confusion,
                'Kappa': kappa,
                'Respuestas reales': respuestas_experto1,
                'Predicciones': respuestas_experto2,
            }
        else:
            resultados_kappa[columna] = None

    promedio_kappa = sum(resultados_kappa[columna]['Kappa'] for columna in resultados_kappa if resultados_kappa[columna] is not None) / len(resultados_kappa)

    return resultados_kappa, promedio_kappa

ruta_experto1 = 'expertoS.csv'
ruta_experto2 = 'ontologia.csv'

resultados, promedio_kappa = calcular_kappa_desde_csv(ruta_experto1, ruta_experto2)

for columna, resultados_columna in resultados.items():
    if resultados_columna is not None:
        print(f"\nResultados para {columna}:")
        print(f"Matriz de Confusión:\n{resultados_columna['Matriz de Confusión']}")
        print(f"Kappa: {resultados_columna['Kappa']}")
        print(f"Respuestas reales: {resultados_columna['Respuestas reales']}")
        print(f"Predicciones: {resultados_columna['Predicciones']}")

print(f"\nPromedio de Kappa: {promedio_kappa}")
