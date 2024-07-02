#ESTE ARCHIVO CONTIENE 3 CODIGOS (1 REDUCE Y MUESTRA COMPARATIVA ENTRE IMAGENES A ELECCION - 2 ARMA UN DATASET .H5 - 3 SEPARA IMAGES Y LABELS EN NPY)
#SE RECOMIENDA COMENTAR UN CODIGO MIENTRAS USAS EL OTRO Y BICEVERSA 
from PIL import Image, ExifTags
import os
import matplotlib.pyplot as plt

def reducir_resolucion(carpeta, factor):
    # Lista de archivos en la carpeta
    archivos = os.listdir(carpeta)

    # Iterar sobre cada archivo en la carpeta
    for archivo in archivos:
        # Ruta completa del archivo
        ruta_completa = os.path.join(carpeta, archivo)

        # Verificar si es un archivo de imagen
        if os.path.isfile(ruta_completa):
            try:
                # Abrir la imagen
                imagen = Image.open(ruta_completa)

                # Intentar obtener los datos EXIF de la imagen
                try:
                    for orientation in ExifTags.TAGS.keys():
                        if ExifTags.TAGS[orientation] == 'Orientation':
                            break
                    exif = dict(imagen._getexif().items())
                    if exif[orientation] == 3:
                        imagen = imagen.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        imagen = imagen.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        imagen = imagen.rotate(90, expand=True)
                except (AttributeError, KeyError, IndexError):
                    # No hay datos EXIF, no se hace nada
                    pass

                # Obtener las dimensiones originales de la imagen
                ancho, alto = imagen.size

                # Calcular las nuevas dimensiones reduciendo a los factores deseados
                nuevo_ancho = int(ancho / factor)
                nuevo_alto = int(alto / factor)

                # Redimensionar la imagen con otro método de interpolación (por ejemplo, BILINEAR)
                imagen_reducida = imagen.resize((nuevo_ancho, nuevo_alto), resample=Image.BILINEAR)

                # Guardar la imagen reducida con un nombre específico
                nombre_reducido = f"{archivo.split('.')[0]}_reducida.jpg"
                ruta_reducida = os.path.join(carpeta, nombre_reducido)
                imagen_reducida.save(ruta_reducida)
                print(f"Se Ha Reducido La Resolución De {archivo} Correctamente.")

            except Exception as e:
                print(f"No Se Pudo Reducir La Resolución De {archivo}: {str(e)}")

def mostrar_comparacion(carpeta, nombre_imagen):
    # Ruta completa de la imagen original y reducida
    ruta_original = os.path.join(carpeta, nombre_imagen)
    ruta_reducida = os.path.join(carpeta, f"{nombre_imagen.split('.')[0]}_reducida.jpg")

    # Verificar si los archivos de imagen reducida y original existen
    if os.path.exists(ruta_original) and os.path.exists(ruta_reducida):
        try:
            # Abrir las imágenes
            imagen_original = Image.open(ruta_original)
            imagen_reducida = Image.open(ruta_reducida)

            # Mostrar la comparación utilizando Matplotlib
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(imagen_original)
            axs[0].set_title('Imagen Original')
            axs[0].axis('off')
            axs[1].imshow(imagen_reducida)
            axs[1].set_title('Imagen Reducida')
            axs[1].axis('off')
            plt.show()
        except Exception as e:
            print(f"No se pudo mostrar la comparación: {str(e)}")
    else:
        print("No se encontraron los archivos de imagen original o reducida.")

# Carpeta que contiene las imágenes
carpeta_imagenes = r"C:/Users/lmichel/Desktop/Fotos_Prediccion"

# Llamar a la función para reducir la resolución de las imágenes en la carpeta
factores = [15]  # CAMBIAR EL NUMERO AL FACTOR DESEADO
for factor in factores:
    reducir_resolucion(carpeta_imagenes, factor)

# Solicitar al usuario el nombre de la imagen que desea visualizar
nombre_imagen = input("Ingrese el nombre de la imagen que desea visualizar (incluyendo la extensión): ")
# Mostrar la comparación entre la imagen original y la reducida
mostrar_comparacion(carpeta_imagenes, nombre_imagen)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

#UNA VEZ SE HAYAN REDUCIDO TODAS LAS IMAGENES COMENTAR EL CODIGO DE ARRIBA Y USAR ESTE CODIGO PARA CREAR UN DATASET .H5
import os
import numpy as np
import h5py
from PIL import Image

# Ruta al directorio del dataset
dataset_dir = 'C:/Users/Leandro/Desktop/DataSet_1a20_800_225x400_RGB'

# Tamaño de las imágenes
img_width, img_height = 225, 400

# Obtener las clases del dataset (subcarpetas)
classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d.isdigit()]
num_classes = len(classes)

# Inicializar listas para almacenar las imágenes y etiquetas
images = []
labels = []

# Procesar cada clase
for class_name in classes:
    class_dir = os.path.join(dataset_dir, class_name)
    # La etiqueta es el nombre de la subcarpeta convertido a entero
    class_idx = int(class_name) - 1  # Restar 1 para que las clases sean de 0 a 19
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = Image.open(img_path).resize((img_width, img_height))
        img_array = np.array(img)
        
        # Asegurarse de que la imagen tiene tres canales (RGB)
        if img_array.shape == (img_height, img_width, 3):
            images.append(img_array)
            labels.append(class_idx)

# Convertir listas a arrays de NumPy
images = np.array(images)
labels = np.array(labels)

# Guardar las imágenes y etiquetas en un archivo HDF5
with h5py.File('DataSet.h5', 'w') as h5f:
    h5f.create_dataset('images', data=images)
    h5f.create_dataset('labels', data=labels)

print("Terminó el proceso")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

#EN EL CASO DE REQUERIR LOS IMAGES Y LABELS EN EXTENSION NPY ESTA PARTE DEL CODIGO ES LA QUE SE DEBE USAR (ESTOS SON LOS ARCHIVOS NECESARIOS PARA CORRER LA RED)
import os
import numpy as np
from PIL import Image

# Ruta al directorio del dataset
dataset_dir = 'D:/Users/Leandro/Downloads/DataSet_1a20_1000_225x400_RGB'

# Tamaño de las imágenes
img_width, img_height = 225, 400

# Obtener las clases del dataset (subcarpetas)
classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d.isdigit()]
num_classes = len(classes)

# Inicializar listas para almacenar las imágenes y etiquetas
images = []
labels = []

# Procesar cada clase
for class_name in classes:
    class_dir = os.path.join(dataset_dir, class_name)
    # La etiqueta es el nombre de la subcarpeta convertido a entero
    class_idx = int(class_name) - 1  # Restar 1 para que las clases sean de 0 a 19
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = Image.open(img_path).resize((img_width, img_height))  # Mantener la imagen en RGB
        img_array = np.array(img)
        
        # Asegurarse de que la imagen tiene tres canales (RGB)
        if img_array.shape == (img_height, img_width, 3):
            images.append(img_array)
            labels.append(class_idx)

# Convertir listas a arrays de NumPy
images = np.array(images)
labels = np.array(labels)

# Guardar las imágenes y etiquetas en archivos .npy
np.save('D:/Users/Leandro/Downloads/DataSet_1a20_1000_225x400_RGB/images.npy', images)
np.save('D:/Users/Leandro/Downloads/DataSet_1a20_1000_225x400_RGB/labels.npy', labels)

print("Terminó el proceso")
