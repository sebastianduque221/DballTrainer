import cv2
import mediapipe as mp
import os
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from detecciones.deteccion_saque import evaluar_saque
from detecciones.deteccion_colocador import detectar_colocador

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def procesar_frame(frame, pose, deteccion_func):
    """Procesa un frame, dibuja los puntos de referencia y evalúa la detección."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Hacer la imagen no escribible para mejorar el rendimiento

    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Inicializar valores por defecto
    angulo_codo = 0
    angulo_rodilla = 0
    angulo_tronco = 0
    manos_sobre_frente = False
    evaluacion = ""

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
        )

        angulo_codo, angulo_rodilla, angulo_tronco, manos_sobre_frente, evaluacion = \
            deteccion_func(results.pose_landmarks.landmark)

        # Convertir a PIL para dibujar texto
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        try:
            font = ImageFont.truetype(r'fonts/static/OpenSans_Condensed-Italic.ttf', 24)
        except IOError:
            font = ImageFont.load_default()

        y0, dy = 30, 30
        for i, line in enumerate(evaluacion.split('\n')):
            y = y0 + i * dy
            draw.text((10, y), line, font=font, fill=(255, 0, 0))

        image = np.array(pil_image)

    return image, results, angulo_codo, angulo_rodilla, angulo_tronco, manos_sobre_frente

def get_unique_filename(directory, base_name, extension):
    """Genera un nombre de archivo único en el directorio especificado."""
    i = 1
    while True:
        filename = f"{base_name}{i}.{extension}"
        if not os.path.exists(os.path.join(directory, filename)):
            return filename
        i += 1

def main():
    # Elegir la detección (saque o colocador)
    deteccion = input("Elige la detección (saque/colocador): ").strip().lower()
    if deteccion == "saque":
        deteccion_func = evaluar_saque
    elif deteccion == "colocador":
        deteccion_func = detectar_colocador
    else:
        print("Detección no válida.")
        return

    # Elegir la fuente de video (video o cámara)
    fuente = input("Elige la fuente de video (video/camara): ").strip().lower()
    if fuente == "video":
        ruta_video = input("Ingresa la ruta del video: ").strip()
        print(f"Ruta del video: {ruta_video}")  # Mensaje de depuración
        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            print(f"No se pudo abrir el video: {ruta_video}")
            return
        base_name = os.path.splitext(os.path.basename(ruta_video))[0]
    elif fuente == "camara":
        cap = cv2.VideoCapture(0)
        base_name = "camara"
    else:
        print("Fuente de video no válida.")
        return

    if not cap.isOpened():
        print("Error al abrir la fuente de video.")
        return

    # Crear la carpeta de salidas si no existe
    output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salidas")
    os.makedirs(output_directory, exist_ok=True)

    # Generar un nombre de archivo único para el CSV de análisis de postura
    csv_filename = get_unique_filename(output_directory, base_name, "csv")
    csv_filepath = os.path.join(output_directory, csv_filename)

    # Crear archivo CSV y escribir encabezados
    with open(csv_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Angulo Codo', 'Angulo Rodilla', 'Angulo Tronco', 'Manos Sobre Frente'])

    frame_number = 0  # Inicializar el número de frame

    # Definir el codec y crear el objeto VideoWriter
    video_output_filename = os.path.join(output_directory, get_unique_filename(output_directory, base_name, "avi"))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Asegurar un valor razonable
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_output_filename, fourcc, fps, (frame_width, frame_height))

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("No se puede recibir el frame. Saliendo...")
                break

            image, results, angulo_codo, angulo_rodilla, angulo_tronco, manos_sobre_frente = procesar_frame(frame, pose, deteccion_func)

            # Mostrar la imagen con los puntos de referencia
            cv2.imshow('Puntos de Referencia', image)

            # Grabar el video
            out.write(image)

            # Guardar datos en el archivo CSV
            if results.pose_landmarks:
                with open(csv_filepath, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([frame_number, angulo_codo, angulo_rodilla, angulo_tronco, manos_sobre_frente])

            frame_number += 1  # Incrementar el número de frame

            if cv2.waitKey(5) & 0xFF == 27:  # Presiona 'Esc' para salir
                break

    cap.release()
    out.release()
    if hasattr(cv2, 'destroyAllWindows'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
