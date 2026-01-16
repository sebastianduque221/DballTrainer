# server.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
import os
import csv
import cv2
import numpy as np
import time
import tempfile
from mediapipe.python.solutions import pose, drawing_utils  # type: ignore

# Importa tus funciones de detección
from detecciones.deteccion_saque import evaluar_saque
from detecciones.deteccion_colocador import detectar_colocador
from main import procesar_frame  # tu función de procesado de cada frame

app = FastAPI()

# --- Funciones auxiliares ---
def get_unique_filename(base="output", ext="csv"):
    """Genera nombre único usando timestamp"""
    ts = int(time.time() * 1000)
    return f"{base}_{ts}.{ext}"

# --- Endpoints ---
@app.get("/")
async def home():
    """Sirve la página HTML de frontend"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...), posicion: str = "saque"):
    # Elegir función de detección
    if posicion == "saque":
        deteccion_func = evaluar_saque
    elif posicion == "colocador":
        deteccion_func = detectar_colocador
    else:
        return {"error": "Posición desconocida"}

    # Crear carpeta de salida
    output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salidas")
    os.makedirs(output_directory, exist_ok=True)

    # Crear nombre de CSV y vídeo de salida
    csv_filename = get_unique_filename("results", "csv")
    csv_filepath = os.path.join(output_directory, csv_filename)

    video_filename = get_unique_filename("video", "avi")
    video_output_path = os.path.join(output_directory, video_filename)

    # Guardar UploadFile en un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    # Abrir el vídeo con OpenCV
    cap = cv2.VideoCapture(tmp_path)

    if not cap.isOpened():
        os.remove(tmp_path)
        return {"error": "No se pudo abrir el vídeo"}

    # Configuración VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

    # Abrir CSV para escritura
    with open(csv_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Angulo Codo', 'Angulo Rodilla', 'Angulo Tronco', 'Manos Sobre Frente'])

        frame_number = 0

        # Procesar cada frame con Mediapipe
        with pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as mp_pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Procesar frame
                image, results, angulo_codo, angulo_rodilla, angulo_tronco, manos_sobre_frente = procesar_frame(
                    frame, mp_pose, deteccion_func
                )

                # Escribir frame procesado en vídeo de salida
                out.write(image)

                # Guardar datos en CSV
                if results.pose_landmarks:
                    writer.writerow([frame_number, angulo_codo, angulo_rodilla, angulo_tronco, manos_sobre_frente])

                frame_number += 1

    # Liberar recursos
    cap.release()
    out.release()
    os.remove(tmp_path)  # borrar archivo temporal

    # Leer CSV final para devolverlo al frontend
    with open(csv_filepath, "r", encoding="utf-8") as f:
        csv_content = f.read()

    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=results.csv"}
    )
