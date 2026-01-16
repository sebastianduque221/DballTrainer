import cv2
import logging
from datetime import datetime
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def procesar_frame(frame, pose, deteccion_func, frame_number):
    """Procesa un frame y evalúa la detección."""
    try:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        evaluacion_resultados = {"mensajes": ["No se detectaron puntos de referencia"], "datos": []}
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = list(results.pose_landmarks.landmark)
            evaluacion_resultados = deteccion_func(landmarks)
        return frame, evaluacion_resultados
    except Exception as e:
        logging.error(f"Error procesando frame en procesamiento.py: {e}")
        return frame, {"mensajes": ["Error en la evaluación"], "datos": []}