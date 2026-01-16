import math
from mediapipe.python.solutions.pose import PoseLandmark
from evaluaciones import evaluar_contacto

def calcular_angulo(p1, p2, p3):
    """Calcula el ángulo entre tres puntos."""
    try:
        angulo = math.degrees(
            math.atan2(p3.y - p2.y, p3.x - p2.x) -
            math.atan2(p1.y - p2.y, p1.x - p2.x)
        )
        return abs(angulo) if angulo >= 0 else abs(angulo + 360)
    except Exception as e:
        print(f"Error al calcular el ángulo: {e}")
        return None

def detectar_remate(landmarks):
    """
    Detecta y evalúa la técnica de remate.
    Args:
        landmarks (list): Lista de landmarks detectados por MediaPipe.
    Returns:
        dict: Resultados de la evaluación con mensajes y datos relevantes.
    """
    try:
        # Validar landmarks
        required_landmarks = [
            PoseLandmark.RIGHT_SHOULDER.value,
            PoseLandmark.RIGHT_ELBOW.value,
            PoseLandmark.RIGHT_WRIST.value,
            PoseLandmark.LEFT_SHOULDER.value,
            PoseLandmark.LEFT_ELBOW.value,
            PoseLandmark.LEFT_WRIST.value
        ]
        if not isinstance(landmarks, list) or not all(idx < len(landmarks) for idx in required_landmarks):
            raise ValueError("Faltan landmarks esenciales para detectar el remate.")

        # Acceder a landmarks individuales
        hombro_der = landmarks[PoseLandmark.RIGHT_SHOULDER.value]
        codo_der = landmarks[PoseLandmark.RIGHT_ELBOW.value]
        muñeca_der = landmarks[PoseLandmark.RIGHT_WRIST.value]

        hombro_izq = landmarks[PoseLandmark.LEFT_SHOULDER.value]
        codo_izq = landmarks[PoseLandmark.LEFT_ELBOW.value]
        muñeca_izq = landmarks[PoseLandmark.LEFT_WRIST.value]

        # Calcular ángulos de los codos
        angulo_codo_der = calcular_angulo(hombro_der, codo_der, muñeca_der)
        angulo_codo_izq = calcular_angulo(hombro_izq, codo_izq, muñeca_izq)

        if angulo_codo_der is None or angulo_codo_izq is None:
            raise ValueError("No se pudieron calcular algunos ángulos.")

        # Evaluar la altura del brazo derecho e izquierdo
        altura_brazo_der_correcta = muñeca_der.y < hombro_der.y
        altura_brazo_izq_correcta = muñeca_izq.y < hombro_izq.y

        # Evaluar el contacto con el balón
        contacto_valido = evaluar_contacto(landmarks)

        # Evaluar si el remate es válido
        remate_valido = (
            contacto_valido["valido"] and
            angulo_codo_der >= 90 and altura_brazo_der_correcta and
            angulo_codo_izq >= 90 and altura_brazo_izq_correcta
        )

        # Mensajes descriptivos
        mensajes = [
            f"Ángulo del codo derecho: {angulo_codo_der:.2f}°",
            f"Ángulo del codo izquierdo: {angulo_codo_izq:.2f}°",
            f"Altura del brazo derecho: {'Correcta' if altura_brazo_der_correcta else 'Incorrecta'}",
            f"Altura del brazo izquierdo: {'Correcta' if altura_brazo_izq_correcta else 'Incorrecta'}",
            contacto_valido["mensaje"],
            f"Remate {'válido' if remate_valido else 'no válido'}"
        ]

        # Salida estructurada
        return {
            "mensajes": mensajes,
            "datos": [
                angulo_codo_der, angulo_codo_izq,
                altura_brazo_der_correcta, altura_brazo_izq_correcta,
                contacto_valido["valido"], remate_valido
            ]
        }

    except Exception as e:
        print(f"Error en detectar_remate: {e}")
        return {
            "mensajes": ["Error en la detección del remate"],
            "datos": [None, None, None, None, None, False]
        }