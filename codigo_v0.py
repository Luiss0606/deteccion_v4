# Importar librerias
import cv2
import time
import numpy as np
import easyocr
from Funciones import *
from datetime import datetime

# Declaracion de variables y constantes
salida_0=[]
salida_1=[]
salida_2=[]
salida_3=[]
salida_4=[]
salida_5=[]

valor_f=""

pos=0

tiempo_bbox=2668269365.4799159

CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.4
COLORS = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (255, 0, 0)]

# Establecer el idioma de la lectura de la placa
tex = easyocr.Reader(['en'], gpu=True)

# Carga las clases de los objetos a detectar (en este caso solo es una clase, placas)
class_names = []
with open("obj.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Cargar el video
cap = cv2.VideoCapture("videoPlacas1.mp4")


# Cargar el modelo de la red neuronal
net = cv2.dnn.readNet("custom-yolov4-tiny-detector_best.weights", "custom-yolov4-tiny-detector.cfg")

# Establecer como backend CUDA para acelerar el procesamiento de la red neuronal
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# Presicion de punto flotante 16 bits
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

# Se crea un modelo de deteccion con la red neuronal
model = cv2.dnn_DetectionModel(net)
# Se establece el tamaño de la imagen de entrada, la escala de los pixeles y el orden de los canales (lo normalizamos)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

# Iniciar el procesamiento de la imagen, bucle principal
while cv2.waitKey(1) < 1:
    # Obtener el frame y verificar que se haya obtenido correctamente
    (grabbed, frame) = cap.read()
    if not grabbed:
        exit()
    
    # Dimensiones del frame
    height, width, _ = frame.shape

    # Inicio del tiempo de cada frame
    start = datetime.utcnow()

    # Deteccion de las placas
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
 
    # Guardar el tiempo de procesamiento del frame
    tiempo_frame=time.time()

    # Iterar sobre las detecciones
    for (classid, score, box) in zip(classes, scores, boxes):
        # Establece f = True para indicar que se detecto una placa
        f = True

        # Dibujar el rectangulo de la placa
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %.2f" % (class_names[0], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (48, 66, 227), 2)
        #print(pos-box[0])
        
        # Si la placa se encuentra a la derecha del frame, se reinician las variables
        if len(salida_0) != 0 and (pos-box[0])>450:
            salida_0 = []
            salida_1 = []
            salida_2 = []
            salida_3 = []
            salida_4 = []
            salida_5 = []
            f = False
            break

        # Establecer el valor de la posicion del box de la placa
        x, y, w, h = box
        crop_img = frame[y:y + h, x:x + w]

        # Procesamiento del box de la placa
        gray = get_grayscale(crop_img)
        gran = get_resize(gray)
        gran = cv2.GaussianBlur(gran,(17,17),0)

        # Mostrar segunda ventana
        cv2.imshow("Crop", cv2.resize(gran, (533,300)))

        # Obtener el texto de la placa
        res = tex.readtext(gran, allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUWXYZV', rotation_info=[0, 5, 10])

        for n in range(len(res)):
            if res[-(n+1)][2] > 0.15:
                text = res[-(n+1)][1]

                if len(text) >= 6:
                    salida_0.append(text[0])
                    salida_1.append(text[1])
                    salida_2.append(text[2])
                    salida_3.append(text[3])
                    salida_4.append(text[4])
                    salida_5.append(text[5])

                    # print(salida_0)
                    # print(salida_1)
                    # print(salida_2)
                    # print(salida_3)
                    # print(salida_4)
                    # print(salida_5)


                    # Obtenemos el valor mas frecuente de cada caracter
                    valor_f = str(most_freq(salida_0)) + str(most_freq(salida_1)) + str(most_freq(salida_2)) + str(
                        most_freq(salida_3)) + str(most_freq(salida_4)) + str(most_freq(salida_5))
                    
                    # Mostrar el valor de la placa en cada frame
                    cv2.putText(frame, valor_f, (box[0], box[1] + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (47, 47, 155), 2)
        
        # aux=box
        pos = box[0]
        tiempo_bbox = time.time()

    if (tiempo_frame - tiempo_bbox) > 2 and f == True:
        now = datetime.now()
        file = open("LOG_Prueba.txt", "a")
        data = [valor_f + ", " + str(now.strftime("%Y-%m-%d %H:%M:%S")) + "\n"]
        file.writelines(data)
        file.close()

        print("Valor final", valor_f)
        f = False

    # Fin del tiempo de procesamiento
    end = datetime.utcnow()

    # Colocamos la informacion de FPS en la imagen
    fps = 'FPS: %.2f ' % (1 / (end - start).total_seconds())
    cv2.putText(frame, fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Cambiar el tamaño de la ventana de salida para mostrar a (1028,720)
    cv2.imshow("output", cv2.resize(frame, (1280, 720)))


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()










