import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time
import os
import matplotlib.pyplot as plt 

def cargar_y_preprocesar_imagen(ruta_imagen, img_height, img_width):
    img = Image.open(ruta_imagen).convert('RGB')  # Convertir la imagen a RGB
    img = img.resize((img_width, img_height))  # Redimensionar la imagen
    img_array = np.array(img)  # Convertir la imagen a un array de NumPy
    img_array = img_array.astype('float32') / 255.0  # Normalizar la imagen
    img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones para que sea (1, img_height, img_width, 3)
    return img_array, img  # Devolver tanto el array como la imagen original

def predecir_clase(modelo, img_array):
    prediccion = modelo.predict(img_array)
    clase_predicha = np.argmax(prediccion, axis=1) + 1  # Obtener el número de maíces predicho
    return clase_predicha[0], prediccion

def mostrar_resultados(img, clase_predicha, prediccion, tiempo_prediccion):
    max_value = np.max(prediccion)
    average_difference = np.mean(np.abs(prediccion - max_value))
    sum_of_elements = np.sum(prediccion)

    print(f"\n-------------------------------")
    print(f'EL NUMERO DE MAICES DETECTADO EN LA IMAGEN ES: {clase_predicha}')
    print(f"-------------------------------")
    print(f"\n-------------------------------\nVECTOR:\n{prediccion}\n-------------------------------")
    print(f'El valor mas alto del vector de prediccion es: {max_value:.4f}')
    print(f'El promedio de diferencia con los demas valores es: {average_difference:.4f}')
    print(f'La suma de todos los valores del vector es: {sum_of_elements:.4f}')
    print(f'\nTiempo de predicción: {tiempo_prediccion:.4f} segundos\n')

# Ruta del modelo guardado y de la carpeta de imágenes
ruta_modelo = 'D:/Users/Leandro/Downloads/Redes Neuronales/Modelos/Modelo_96.5%_1000f_RGB_Maices_Solos.h5'
ruta_carpeta_imagenes = 'D:/Users/Leandro/Downloads/Redes Neuronales/Fotos_Prediccion/Prueba'

# Dimensiones de la imagen (ajusta según tu modelo)
img_height, img_width = 400, 225

# Cargar el modelo
modelo = load_model(ruta_modelo)

# Listar todos los archivos en la carpeta de imágenes
imagenes = os.listdir(ruta_carpeta_imagenes)

# Lista para almacenar los valores máximos de cada predicción
max_values_list = []

# Lista para almacenar los nombres de las imágenes clasificadas incorrectamente
incorrectas = []

# Contadores para calcular la precisión
total_imagenes = 0
correctas = 0

for imagen in imagenes:
    # Obtener el número de maíces real del nombre del archivo
    numero_maices_real = int(imagen.split('_')[0])  # Obtener el número de maíces del nombre del archivo

    # Cargar y preprocesar la imagen para la predicción
    img_array, img_original = cargar_y_preprocesar_imagen(os.path.join(ruta_carpeta_imagenes, imagen), img_height, img_width)

    # Realizar la predicción
    start_time = time.time()
    clase_predicha, prediccion = predecir_clase(modelo, img_array)
    end_time = time.time()
    prediction_time = end_time - start_time

    # Mostrar resultados de la predicción
    mostrar_resultados(img_original, clase_predicha, prediccion, prediction_time)

    # Guardar el valor máximo del vector de predicción
    max_values_list.append(np.max(prediccion))

    # Contabilizar la precisión
    total_imagenes += 1
    if clase_predicha == numero_maices_real:
        correctas += 1
    else:
        incorrectas.append(imagen)  # Agregar nombre de imagen incorrecta

# Calcular el valor promedio entre los máximos valores de los vectores
average_max_value = np.mean(max_values_list)
print(f"-------------------------------")
print(f'El valor promedio entre los máximos valores de los vectores es: {average_max_value:.4f}\n-------------------------------\n')

# Calcular la precisión
precision = correctas / total_imagenes if total_imagenes > 0 else 0
precision_porcentaje = precision * 100
print(f"-------------------------------")
print(f'La precisión del modelo es: {precision:.4f} ({precision_porcentaje:.2f}%)')
print(f'El número total de imágenes evaluadas es: {total_imagenes}')
print(f'El número total de imágenes clasificadas correctamente es: {correctas}')
print(f'El número total de imágenes clasificadas incorrectamente es: {total_imagenes - correctas}')
print(f"-------------------------------\n")

# Mostrar nombres de imágenes clasificadas incorrectamente
print("Resumen de imágenes clasificadas incorrectamente:")
for imagen in incorrectas:
    print(imagen)

# Preguntar al usuario si desea visualizar alguna imagen
opcion = input("\n¿Desea visualizar alguna imagen? Ingrese el nombre y extensión ('salir' para terminar): ")

while opcion.lower() != 'salir':
    if opcion in incorrectas:
        # Cargar y mostrar la imagen seleccionada
        ruta_imagen_seleccionada = os.path.join(ruta_carpeta_imagenes, opcion)
        img_array, img_original = cargar_y_preprocesar_imagen(ruta_imagen_seleccionada, img_height, img_width)
        clase_predicha, prediccion = predecir_clase(modelo, img_array)
        
        # Mostrar la imagen
        plt.figure(figsize=(6, 6))
        plt.imshow(img_original)
        plt.title(f'Imagen: {opcion}\nClase Real: {int(opcion.split("_")[0])} - Clase Predicha: {clase_predicha}')
        plt.axis('off')
        plt.show()
    else:
        print(f"La imagen '{opcion}' no está en la lista de imágenes clasificadas incorrectamente.")
    
    # Preguntar nuevamente si desea visualizar otra imagen
    opcion = input("\n¿Desea visualizar otra imagen? Ingrese el nombre y extensión ('salir' para terminar): ")

print("\nTerminó la ejecución de la red")





