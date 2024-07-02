![](img/GFA-logo1.png)

# Contando Objetos Con Redes Neuronales Convolucionales 

Este trabajo fue llevado por Ivan Leiva y Michel Leandro dentro de las Practicas profesionalizantes de la escuela de educacion tecnica N°1 "Luciano Reyes", guiado Cerrotta Santiago de parte del laboratorio de fotonica aplicada de la UTN-FRD durante 200 horas entre marzo y julio de 2024.

### 1. Objetivo

Crear un dataset con imagenes propias para entrenar una red neuronal. A partir de este, diseñar y optimizar una arquitectura de una red convolucional para clasificar la cantidad de maices (1 a 20).

### 2. Introducción

Las Redes Neuronales Convolucionales (CNN) son un tipo de red neuronal diseñada para procesar y analizar datos con una estructura de cuadrícula, como imágenes. Han revolucionado el reconocimiento de imágenes y tienen aplicaciones en video, procesamiento del lenguaje natural y bioinformática.

1. Componentes Principales
Convolución (Filtros y Kernel)

➤Filtros (Kernels): Son matrices pequeñas que recorren la imagen para detectar características específicas como bordes, texturas y patrones. Cada filtro se aplica a una región de la imagen de entrada y produce una característica en la salida.

➤Operación de Convolución: Multiplica los valores del kernel por los valores de la imagen en la región cubierta por el kernel y suma estos valores para obtener un solo valor en la salida. Esto se repite para cada posición del kernel en la imagen.

<div align="center">
	<img src="/img/convolucion.gif" width="480" height="380">
</div>
<div align="center">
	<em> Figura 1 - Convolucion </em>
</div>

2. Mapas de Características (Feature Maps)

➤Generación de Características: Al aplicar múltiples filtros a una imagen de entrada, se generan varios mapas de características que capturan diferentes aspectos de la imagen.

➤Dimensiones:
Alto y Largo: Dependen del tamaño del filtro, el paso (stride) y el relleno (padding). No siempre son iguales a las dimensiones de la imagen de entrada.
Profundidad: Igual al número de filtros aplicados en la capa de convolución.

<div align="center">
	<img src="/img/cifar-10.png">
	<em> Figura 2 - Esquema Feature Map de CIFAR-10</em>
</div>

3. Pooling

➤Max Pooling y Average Pooling: Reducen la dimensionalidad de los mapas de características. Max pooling toma el valor máximo en una región específica, mientras que average pooling toma el promedio. Esto reduce el número de parámetros y la carga computacional, además de hacer la red más robusta a pequeñas variaciones en la posición de las características.

<div align="center">
	<img src="/img/maxpool.gif" width="580" height="380">
</div>
<div align="center">
	<em> Figura 3 - MaxPool </em>
</div>

4. Capas Completamente Conectadas (Fully Connected Layers)

➤Clasificación: Después de varias capas convolucionales y de pooling, los mapas de características se aplanan y pasan a través de una o más capas completamente conectadas. Estas capas actúan como una red neuronal tradicional y se utilizan para la clasificación final.

<div align="center">
	<img src="/img/fullyconect.gif" width="480" height="380">
</div>
<div align="center">
	<em> Figura 4 - Fully Conect Layers </em>
</div>

5. Algoritmos de Entrenamiento

➤Backpropagation: Ajusta los pesos de la red calculando el gradiente del error con respecto a cada peso mediante descenso de gradiente.

➤Optimización: Algoritmos como SGD (Stochastic Gradient Descent), Adam y RMSprop actualizan los pesos de manera eficiente durante el entrenamiento.

➤Función de Pérdida: Las CNNs usan funciones de pérdida como la entropía cruzada para cuantificar el error entre las predicciones de la red y las etiquetas reales.

6. Ventajas de las CNN

➤Extracción Automática de Características: Los filtros aprenden automáticamente a detectar características relevantes durante el entrenamiento.

➤Invariancia a la Translación: Las operaciones de pooling hacen que las CNNs sean robustas a la posición de las características dentro de la imagen.

➤Reducción de Parámetros: Las CNNs reducen significativamente el número de parámetros gracias al uso de filtros compartidos y pooling.

7. Aplicaciones

➤Reconocimiento de Imágenes: Clasificación, detección de objetos, segmentación semántica.

➤Procesamiento de Video: Detección y seguimiento de objetos en secuencias de video.

➤Procesamiento de Lenguaje Natural: Análisis de sentimientos, clasificación de texto.

➤Bioinformática: Análisis de secuencias de ADN, predicción de estructuras proteicas.
	
### 3. Armado Del Dataset 

1. Se armo el siguiente dispositivo para tomar las fotos.

<div align="center">
	<img src="/img/DispositivoSacadoDeFotos.png">
</div>
<div align="center">
	<center><em>Figura 5 - Dispositivo de Toma de Fotos</em></center>
</div>

2. Criterios para capturar las fotos.

➤Fondo de cartulina negra (22cm de ancho x 27cm de largo con un margen de 1cm).

➤Altura de la camara 18cm de alto.

➤Celular Samsung A21s.

➤Mover los objetos con un palito de crochet para que cada foto sea diferente.

➤Se tomaron 1000 fotos de maices solos. (50 fotos por cantidad de maices de 1 a 20 maices).

➤Se tomaron 2000 fotos de maíces con lentejas. Se capturaron 100 fotos por cada cantidad de maíz de 1 a 10 lentejas, comenzando con 1 maíz y 1 lenteja y aumentando hasta 1 maíz y 10 lentejas. Luego, el número de maíces se incrementó de 1 a 20, manteniendo el mismo rango de lentejas de 1 a 10, tomando 10 fotos por cada cantidad de maices con 1 lenteja y asi hasta 10 con un total de 100 fotos por cantidad de maices.

➤Se tomaron 2000 fotos de maices con lenteas con arroz. Con los mismos criterios que maices con lentejas pero con un fondo de arroz.

➤Caracteristicas de las fotos (2250x4000).

<div align="center">
	<img src="/img/FotoPura.png">
</div>
<div align="center">
	<em> Figura 6 - Foto Pura De Ejemplo De Los 3 Datasets Armados </em>
</div>

3. Criterios para el armado del dataset.

➤Se redujo la resolucion de las fotos un factor 10 para que tuvieran un tamaño menor sin perder tanta informacion de la misma..

➤Caracteristicas de las fotos reducidas (225x400).

<div align="center">
	<img src="/img/FotoReducida.png">
</div>
<div align="center">
	<em> Figura 7 - Foto Reducida De Ejemplo De Los 3 Datasets Armados</em>
</div>

➤Mayormente se tomo de ejemplo el dataset CIFAR-10.

➤Se renombraron las fotos para un mejor orden segun su numero de foto y cantidad de maices (NumeroDeFoto_CantidadDeMaices ejemplo 1_1, 1_2 y asi hasta 50_1 y luego con 2 maices hasta 20). En el caso de lentejas seria cantidad de maices, cantidad de lentejas y numero de foto (1_1_1 hasta 1_10_10 hasta 20 maices.) y con arroz seria cantidad de maices, cantidad de lentejas, A y numero de foto (1_1_A_1 hasta 1_10_A_10 hasta 20 maices.).

➤Se ordenaron dentro de una carpeta llamada DataSet_1a20_NumeroDeImagenes_Maices_(Solos o Acompañamiento)_225x400_RGB.
La misma tiene subcarpetas del 1 al 20 donde dentro se ordenaron las fotos segun su cantidad de maices. (Fotos de 1 maiz en carpeta 1 y asi sucesivamente).

_[Datasets](https://github.com/Leandrituw/Contando-Objetos-Con-Redes-Neuronales-Convolucionales/tree/main/Armado_Del_Dataset)_

➤Para almacenar los datasets y cargarlos en python los transformamos a archivos con la extension .npy que basicamente lo que hace es leer los archivos de la carpeta DataSet_1a20_NumeroDeImagenes_Maices_(Solos o Acompañamiento)_225x400_RGB y almacenar en una lista el numero de subcarpeta como etiqueta y dentro de la etiqueta guarda la imagen como lista.
¿Porque? Son faciles de cargar y leer en python y tiene un tamaño mucho menor que el dataset puro.

<div align="center">
	<img src="/img/Dataset.png">
</div>
<div align="center">
	<em> Figura 7 - Dataset </em>
</div>

_[Codigo_Armado_Del_Dataset](https://github.com/Leandrituw/Contando-Objetos-Con-Redes-Neuronales-Convolucionales/blob/main/Armado_Del_Dataset/Reducir_Resolucion_Crear_DataSet_RGB.py)_

### 5. Diseño De La Red

➤Librerias utilizadas en el codigo: Numpy, Seaborn, Matplotlib, scikit-learn, TensorFlow Keras, time.

➤Se comenzo con el dataset de solo maices partiendo el mismo en 80% entrenamiento y 20% testeo variando criterios como capas, filtros, pooling, epochs entre otros menos impactantes pero importantes como batch size, dropouts y random state mientras se controlaba la precision de los datos de testeo para conseguir la arquitectura que mejor precision nos diera, una vez conseguida se intento mejorar pero haciendo cambios solo se logro disminuir la precision, el mejor modelo fue guardado con la extension .h5 porque es una extension mas universal que se puede usar fuera del entorno de keras.

➤Se intento mejorar la mejor arquitectura pero poniendo mas o menos capas y filtros lo unico que se logro fue disminuir la precision asi que se definio la siguiente arquitectura ya que con pocas capas y filtros era lo justo y necesario para una precision alta en poco tiempo.

<div align="center">
	<img src="/img/Esquema1.png">
</div>
<div align="center">
	<em> Figura 8 - Arquitectura Final </em>
</div>

➤Todo esto nos dio como resultado el siguiente summary.

<div align="center">
	<img src="/img/Summary.png">
</div>
<div align="center">
	<em> Figura 9 - Summary </em>
</div>

_[Codigo_Entrenar_Red_Neuronal](https://github.com/Leandrituw/Contando-Objetos-Con-Redes-Neuronales-Convolucionales/blob/main/Entrenamiento_De_Arquitecturas/Entrenar_Red_Neuronal_RGB.py)_

### 6. Prediccion y Resultados 

➤Se hizo un codigo en el que se sube una carpeta con imagenes a predecir y el modelo a usar.
Al final se indica cantidad de maices predecida para cada imagen, precision de prediccion, numero de imagenes predecidas correcta e incorrectamente y nombre de imagenes predecidas incorrectamente.

➤¿Como se predicen las imagenes? Al cargarse cada imagen hay un vector de 1 a 20 en el cual todos los valores son 0 y se reemplazan los valores por la probabilidad de que en esa posicion haya esa cantidad de maices, entonces el numero mas cercano a 1 en el vector seria el numero de maices que hay en la foto.
Luego se compara la prediccion con la etiqueta verdadera y se suman las que dio correctamente, finalmente se dividen las predicciones correctas por el total de imagenes y eso nos da la precision.

➤Al entrenar varios datasets juntos, se obtenian buenas precisiones en mas de un dataset ya que parecia ser que la mejor arquitectura para maices solos funcionaba muy bien para los demas datasets.

1. MAICES SOLOS

➤Precision de entrenamiento:

<div align="center">
	<img src="/img/Curva1.png">
</div>
<div align="center">
	<em> Figura 10 - Curva Aprendizaje </em>
</div>

<div align="center">
	<img src="/img/Matriz1.png">
</div>
<div align="center">
	<em> Figura 11 - Matriz Confusion </em>
</div>

➤Precision de prediccion de imagenes:

<div align="center">
	<img src="/img/Predecida1.png">
</div>
<div align="center">
	<em> Figura 12 - Imagen Predecida Maices Solos </em>
</div>

2. MAICES CON LENTEJAS

➤Precision de entrenamiento:

<div align="center">
	<img src="/img/Curva2.png">
</div>
<div align="center">
	<em> Figura 13 - Curva Aprendizaje </em>
</div>

<div align="center">
	<img src="/img/Matriz2.png">
</div>
<div align="center">
	<em> Figura 14 - Matriz Confusion </em>
</div>

➤Precision de prediccion de imagenes:

<div align="center">
	<img src="/img/Predecida2.png">
</div>
<div align="center">
	<em> Figura 15 - Imagen Predecida Maices Con Lentejas </em>
</div>

3. MAICES CON LENTEJAS CON ARROZ

➤En proceso...

_[Codigo_Prediccion](https://github.com/Leandrituw/Contando-Objetos-Con-Redes-Neuronales-Convolucionales/blob/main/Prediccion_De_Imagenes/Prediccion_RGB.py)_

### 7. Recursos 

Dentro Del Repositorio Se Encuentra: 
* 📄Informe-Redes Neuronales Convolucionales - GFA-UTN-FRD📄
* 📂Armado_Del_Dataset📂
* 🡪DataSet_1a20_1000_Maices_Solos_225x400_RGB.zip🡨
* 🡪DataSet_1a20_2000_Maices_Lentejas_225x400_RGB.zip🡨
* 🡪DataSet_1a10_1000_Maices_Lentejas_Arroz_225x400_RGB.zip🡨
* 🡪DataSet_11a20_1000_Maices_Lentejas_Arroz_225x400_RGB.zip🡨
* 🡪Reducir_Resolucion_Crear_DataSet_RGB.py🡨
* 📂Entrenamiento_De_Arquitecturas📂 
* 🡪Entrenar_Red_Neuronal_RGB.py🡨
* 📂Prediccion_De_Imagenes📂 
* 🡪Prediccion_RGB.py🡨

* ⚠️SE RECOMIENDA LEER LOS COMENTARIOS DE LOS CODIGOS⚠️

### 8. Fuentes

_[Playlist](https://www.youtube.com/playlist?list=PL-Ogd76BhmcC_E2RjgIIJZd1DQdYHcVf0)_

_[Blog Red Neuronal Para Detectar Diabetes](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)_

_[Introduccion Redes Convolucionales](https://bootcampai.medium.com/redes-neuronales-convolucionales-5e0ce960caf8)_

_[GitHub Sobre CNN De La UTN-GFA](https://github.com/UTN-GFA/UTN-GFA.github.io)_

_[GitHub Sobre CIFAR-10](https://gist.github.com/eblancoh/d379d92a3680360857581d8937ef114b)_

_[Como Entrenar Una Red Con CIFAR-10](https://datasmarts.net/es/como-entrenar-una-red-neuronal-en-cifar-10-con-keras/)_

_[Blog De Funcionamiento De CIFAR-10/100](https://www.cs.toronto.edu/%7Ekriz/cifar.html)_

_[Como Crear Un Dataset Similar a CIFAR-10](https://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10)_
