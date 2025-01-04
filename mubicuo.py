import tkinter as tk
import subprocess

# Crear la ventana principal
root = tk.Tk()
root.title("Modelo Ubicuo de Análisis Emocional")
root.geometry("500x300")  # Tamaño de la ventana

# Función para el botón Action_Tendency
def action_tendency():
    instructions_text = (
        "Para obtener el comportamiento Action_Tendency:\n\n"
        "1. Analizar uno a uno los videos de los estudiantes.\n"
        "2. Ejecutar el algoritmo Kmeans.\n"
        "3. Obtener el Action_Tendency con la ontología."
    )
    instructions_label.config(text=instructions_text)
    ruta_programa = "D:/programa/autoridad.py"
    
    # Ejecutar el programa autoridad.py
    try:
        subprocess.run(
            ["C:/Users/Susana Arias/AppData/Local/Programs/Python/Python311/python", ruta_programa],
            check=True
        )
    except FileNotFoundError:
        instructions_label.config(text="Error: Archivo autoridad.py no encontrado.")
    except Exception as e:
        instructions_label.config(text=f"Error al ejecutar: {e}")

# Función para el botón Pareto
def pareto():
    instructions_text = (
        "Para obtener el diagrama de Pareto:\n\n"
        "1. Analizar las matrices de emociones para todos los estudiantes.\n"
        "2. Obtener la gráfica.\n"
        "3. Verificar los resultados."
    )
    instructions_label.config(text=instructions_text)
    ruta_programa = "D:/programa/Pareto/Pareto.py"
    
    # Ejecutar el programa Pareto.py
    try:
        subprocess.run(
            ["C:/Users/Susana Arias/AppData/Local/Programs/Python/Python311/python", ruta_programa],
            check=True
        )
    except FileNotFoundError:
        instructions_label.config(text="Error: Archivo Pareto.py no encontrado.")
    except Exception as e:
        instructions_label.config(text=f"Error al ejecutar: {e}")

# Título en la parte superior
title_label = tk.Label(root, text="Modelo Ubicuo de Análisis Emocional Para Clases en Linea", font=("Arial", 12), wraplength=400, justify="center")
title_label.pack(pady=10)

# Contenedor para los botones
frame = tk.Frame(root)
frame.pack(expand=True)

# Botón Action_Tendency
action_tendency_button = tk.Button(frame, text="Action-Tendency", command=action_tendency, width=15)
action_tendency_button.grid(row=0, column=0, padx=20, pady=20)

# Botón Pareto
pareto_button = tk.Button(frame, text="Pareto", command=pareto, width=15)
pareto_button.grid(row=0, column=1, padx=20, pady=20)

# Etiqueta para mostrar instrucciones
instructions_label = tk.Label(root, text="", justify="left", wraplength=400)
instructions_label.pack(side="top", pady=10)

# Pie de página
footer_label = tk.Label(root, text="By: Susana Arias - UNIR", font=("Arial", 10, "italic"))
footer_label.pack(side="bottom", pady=10)

# Iniciar el bucle principal
root.mainloop()

