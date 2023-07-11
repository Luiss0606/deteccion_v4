import cv2
import time
import numpy as np
import easyocr
from Funciones import *
from datetime import datetime
import mysql.connector

# Constantes
CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.4
COLORS = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (255, 0, 0)]
OUTPUT_SIZE = (1280, 720)
CROP_OUTPUT_SIZE = (533, 300)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_COLOR = (48, 66, 227)

# Configuraciones y parámetros
config = {
    "input_size": (416, 416),
    "scale": 1 / 255,
    "swap_rb": True,
    "video_path": "videoPlacas1.mp4",
    "weights_path": "custom-yolov4-tiny-detector_best.weights",
    "cfg_path": "custom-yolov4-tiny-detector.cfg",
    "names_path": "obj.names",
    "db_config": {
        "host": "localhost",
        "user": "root",
        "password": "david12341",
        "database": "seguridad"
    }
}


def load_model(config):
    net = cv2.dnn.readNet(config["weights_path"], config["cfg_path"])
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=config["input_size"], scale=config["scale"], swapRB=config["swap_rb"])
    return model


def load_class_names(names_path):
    with open(names_path, "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]
    return class_names

# Funcion que nos permite verificar la existencia de una placa en la base de datos
def verificar_placa(placa):
    # Configuraciones de la base de datos
    db_config = config["db_config"]

    # Establecer conexión con la base de datos
    connection = mysql.connector.connect(
        host=db_config["host"],
        user=db_config["user"],
        password=db_config["password"],
        database=db_config["database"]
    )

    cursor = connection.cursor()

    # Verificar si la placa existe en la base de datos
    query = "SELECT * FROM placas WHERE n_placa = %s"
    cursor.execute(query, (placa,))
    result = cursor.fetchone()

    # Cerrar el cursor y la conexión
    cursor.close()
    connection.close()

    return result is not None

# Funcion para registrar la placa en la base de datos
def registrar_placa(placa):
    # Configuraciones de la base de datos
    db_config = config["db_config"]

    # Establecer conexión con la base de datos
    connection = mysql.connector.connect(
        host=db_config["host"],
        user=db_config["user"],
        password=db_config["password"],
        database=db_config["database"]
    )

    cursor = connection.cursor()

    # Encontrar el id más pequeño vacío
    query_find = "SELECT id FROM placas WHERE id >= 1 ORDER BY id"
    cursor.execute(query_find)
    ids = [id[0] for id in cursor.fetchall()]

    next_id = next((x for x, y in enumerate(ids, 1) if x != y), len(ids)+1)

    # Insertar la placa en la base de datos en el id vacío más pequeño
    query_insert = "INSERT INTO placas (id, n_placa) VALUES (%s, %s)"
    cursor.execute(query_insert, (next_id, placa))

    # Confirmar los cambios realizados en la base de datos
    connection.commit()

    # Cerrar el cursor y la conexión
    cursor.close()
    connection.close()



def main(config):
    # Inicialización
    tex = easyocr.Reader(['en'], gpu=True)
    class_names = load_class_names(config["names_path"])
    cap = cv2.VideoCapture(config["video_path"])
    model = load_model(config)

    # Variables
    salida = [[] for _ in range(6)]
    valor_f = ""
    pos = 0
    tiempo_bbox = 2668269365.4799159

    while cv2.waitKey(1) < 1:
        # Procesamiento de imagen
        grabbed, frame = cap.read()
        if not grabbed:
            exit()

        start = datetime.utcnow()

        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        tiempo_frame = time.time()

        for (classid, score, box) in zip(classes, scores, boxes):
            f = True

            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %.2f" % (class_names[0], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 5), FONT, FONT_SCALE, FONT_COLOR, 2)

            if any(salida) and (pos - box[0]) > 450:
                salida = [[] for _ in range(6)]
                f = False
                break

            x, y, w, h = box
            crop_img = frame[y:y + h, x:x + w]

            gray = get_grayscale(crop_img)
            gran = get_resize(gray)
            gran = cv2.GaussianBlur(gran, (17, 17), 0)

            cv2.imshow("Crop", cv2.resize(gran, CROP_OUTPUT_SIZE))

            res = tex.readtext(gran, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUWXYZV', rotation_info=[0, 5, 10])

            for n in range(len(res)):
                if res[-(n + 1)][2] > 0.15:
                    text = res[-(n + 1)][1]

                    if len(text) >= 6:
                        for i in range(6):
                            salida[i].append(text[i])

                        valor_f = ''.join([most_freq(s) for s in salida])
                        cv2.putText(frame, valor_f, (box[0], box[1] + 110), FONT, FONT_SCALE, FONT_COLOR, 2)

            pos = box[0]
            tiempo_bbox = time.time()

        if (tiempo_frame - tiempo_bbox) > 2 and f:
            now = datetime.now()

            if not verificar_placa(valor_f):
                print("La placa no existe en la base de datos")
                registrar_placa(valor_f)
                print("La placa", valor_f ," se ha registrado en la base de datos")

            else:
                print("La placa", valor_f ," ya existe en la base de datos")
            with open("LOG_Prueba.txt", "a") as file:
                data = [valor_f + ", " + str(now.strftime("%Y-%m-%d %H:%M:%S")) + "\n"]
                file.writelines(data)

            print("Valor final", valor_f)
            f = False

        end = datetime.utcnow()
        fps = 'FPS: %.2f ' % (1 / (end - start).total_seconds())
        cv2.putText(frame, fps, (0, 25), FONT, 1, (255, 0, 255), 2)

        cv2.imshow("output", cv2.resize(frame, OUTPUT_SIZE))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(config)
