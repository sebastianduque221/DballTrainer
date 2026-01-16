import math
import logging
from mediapipe.python.solutions.pose import PoseLandmark
from evaluaciones import evaluar_sentadillas, evaluar_contacto, evaluar_posicion, evaluar_movimiento

logging.basicConfig(level=logging.ERROR)

def calcular_angulo(p1, p2, p3):
    """Calcula el ángulo entre tres puntos en 2D."""
    try:
        angulo = math.degrees(
            math.atan2(p3.y - p2.y, p3.x - p2.x) -
            math.atan2(p1.y - p2.y, p1.x - p2.x)
        )
        return abs(angulo) if angulo >= 0 else abs(angulo + 360)
    except Exception as e:
        logging.error(f"Error al calcular el ángulo: {e}")
        return None

def calcular_distancia(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    try:
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    except Exception as e:
        logging.error(f"Error al calcular la distancia: {e}")
        return None

def detectar_recibo(landmarks):
    """
    Detecta la postura de recibo basado en los puntos de referencia del cuerpo en voleibol.
    Args:
        landmarks (list): Lista de landmarks detectados por MediaPipe.
    Returns:
        dict: Resultados de la evaluación con mensajes y datos relevantes.
              La lista "datos" contiene 6 elementos:
                [Ángulo del Tronco, Profundidad de Sentadilla, Posición Correcta, 
                 Contacto de Brazos, Movimiento Controlado, Distancia Entre Pies]
    """
    try:
        # Validar landmarks
        required_landmarks = [
            PoseLandmark.LEFT_HIP.value,
            PoseLandmark.LEFT_KNEE.value,
            PoseLandmark.LEFT_ANKLE.value,
            PoseLandmark.LEFT_SHOULDER.value,
            PoseLandmark.NOSE.value,
            PoseLandmark.LEFT_WRIST.value,
            PoseLandmark.RIGHT_WRIST.value,
            PoseLandmark.LEFT_ANKLE.value,  # Puede ser redundante, verificar duplicidad
            PoseLandmark.RIGHT_ANKLE.value
        ]
        if not isinstance(landmarks, list) or not all(idx < len(landmarks) for idx in required_landmarks):
            raise ValueError("Faltan landmarks esenciales para evaluar el recibo.")

        # Landmarks clave
        cadera = landmarks[PoseLandmark.LEFT_HIP.value]
        rodilla = landmarks[PoseLandmark.LEFT_KNEE.value]
        tobillo = landmarks[PoseLandmark.LEFT_ANKLE.value]
        hombro = landmarks[PoseLandmark.LEFT_SHOULDER.value]
        cabeza = landmarks[PoseLandmark.NOSE.value]
        manos = [landmarks[PoseLandmark.LEFT_WRIST.value], landmarks[PoseLandmark.RIGHT_WRIST.value]]
        tobillo_izq = landmarks[PoseLandmark.LEFT_ANKLE.value]
        tobillo_der = landmarks[PoseLandmark.RIGHT_ANKLE.value]

        # Cálculo de ángulos
        angulo_tronco = calcular_angulo(cadera, hombro, cabeza)
        profundidad_sentadilla = calcular_angulo(cadera, rodilla, tobillo)

        # Validación de ángulos
        if angulo_tronco is None or profundidad_sentadilla is None:
            raise ValueError("No se pudieron calcular algunos ángulos.")

        # Evaluaciones
        posicion_correcta = evaluar_posicion(landmarks)
        contacto_brazos = evaluar_contacto(manos)
        sentadilla_correcta = evaluar_sentadillas(profundidad_sentadilla)
        movimiento_excesivo = evaluar_movimiento(landmarks)

        # Calcular distancia entre los pies
        distancia_pies = calcular_distancia(tobillo_izq, tobillo_der)

        # Mensajes descriptivos
        mensajes = [
            f"Ángulo del tronco: {angulo_tronco:.2f}° {'Adecuado' if 70 <= angulo_tronco <= 110 else 'Inadecuado'}",
            f"Profundidad de sentadilla: {'Correcta' if sentadilla_correcta else 'Incorrecta'}",
            f"Posición corporal: {'Adecuada' if posicion_correcta else 'Inadecuada'}",
            f"Contacto con el balón: {'Correcto' if contacto_brazos else 'Incorrecto'}",
            f"Movimiento: {'Controlado' if not movimiento_excesivo else 'Excesivo'}",
            f"Distancia entre pies: {distancia_pies:.2f}"
        ]

        # Salida estructurada con 6 columnas en "datos"
        return {
            "mensajes": mensajes,
            "datos": [
                angulo_tronco,
                profundidad_sentadilla,
                posicion_correcta,
                contacto_brazos,
                not movimiento_excesivo,  # Indica "Movimiento Controlado" (True si es controlado)
                distancia_pies
            ]
        }

    except Exception as e:
        logging.error(f"Error en detectar_recibo: {e}", exc_info=True)
        return {
            "mensajes": ["Error en la detección del recibo"],
            "datos": [None, None, None, None, None, None]
        }

class ReciboEvaluator:
    def evaluate(self, landmarks):
        # Implementa aquí la lógica real de evaluación del recibo
        return {
            "mensajes": ["Evaluación de recibo no implementada."],
            "datos": [None] * 8  # Ajusta la cantidad de datos según tus encabezados
        }

def obtener_encabezados_recibo():
    return [
        "Angulo Tronco", "Profundidad Sentadilla", "Distancia Entre Pies",
        "Simetria", "Recibo Valido", "Contacto", "Estabilidad", "Movimiento"
    ]