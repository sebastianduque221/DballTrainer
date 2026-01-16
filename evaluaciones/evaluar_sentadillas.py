import math
from mediapipe.python.solutions.pose import PoseLandmark
from typing import List, Dict, Union, Tuple


def calcular_angulo(p1, p2, p3) -> Union[float, None]:
    """
    Calcula el ángulo entre tres puntos.
    Args:
        p1, p2, p3: Puntos con atributos x, y.
    Returns:
        float: Ángulo en grados o None si ocurre un error.
    """
    try:
        delta_y1, delta_x1 = p1.y - p2.y, p1.x - p2.x
        delta_y2, delta_x2 = p3.y - p2.y, p3.x - p2.x
        angulo = math.degrees(math.atan2(delta_y2, delta_x2) - math.atan2(delta_y1, delta_x1))
        return angulo % 360  # Asegurar que el ángulo esté en el rango [0, 360)
    except Exception as e:
        print(f"Error al calcular el ángulo: {e}")
        return None


def validar_landmarks(landmarks: List, indices: List[int]) -> bool:
    """
    Valida que los landmarks sean una lista y contengan los índices necesarios.
    Args:
        landmarks (list): Lista de landmarks detectados.
        indices (list): Índices de landmarks requeridos.
    Returns:
        bool: True si los landmarks son válidos, False en caso contrario.
    """
    if not isinstance(landmarks, list) or not all(0 <= idx < len(landmarks) for idx in indices):
        return False
    return True


def evaluar_sentadilla_pierna(cadera, rodilla, tobillo, angulo_min: float, angulo_max: float) -> Tuple[float, bool]:
    """
    Evalúa la sentadilla para una pierna específica.
    Args:
        cadera, rodilla, tobillo: Landmarks de la pierna.
        angulo_min (float): Ángulo mínimo para considerar la sentadilla válida.
        angulo_max (float): Ángulo máximo para considerar la sentadilla válida.
    Returns:
        Tuple[float, bool]: Ángulo de la rodilla y si la sentadilla es válida.
    """
    angulo_rodilla = calcular_angulo(cadera, rodilla, tobillo)
    if angulo_rodilla is None:
        raise ValueError("No se pudo calcular el ángulo de la rodilla.")
    sentadilla_valida = angulo_min <= angulo_rodilla <= angulo_max
    return angulo_rodilla, sentadilla_valida


def evaluar_sentadillas(
    landmarks: List,
    angulo_min: float = 90.0,
    angulo_max: float = 140.0
) -> Dict[str, Union[List[str], List[Tuple[float, bool]]]]:
    """
    Evalúa la técnica de sentadilla basada en los landmarks proporcionados.
    Args:
        landmarks (list): Lista de landmarks detectados por MediaPipe.
        angulo_min (float): Ángulo mínimo para considerar la sentadilla válida.
        angulo_max (float): Ángulo máximo para considerar la sentadilla válida.
    Returns:
        dict: Resultados de la evaluación con mensajes y datos relevantes.
    """
    try:
        # Validar landmarks
        indices_necesarios = [
            PoseLandmark.LEFT_HIP.value,
            PoseLandmark.LEFT_KNEE.value,
            PoseLandmark.LEFT_ANKLE.value,
            PoseLandmark.RIGHT_HIP.value,
            PoseLandmark.RIGHT_KNEE.value,
            PoseLandmark.RIGHT_ANKLE.value,
        ]
        if not validar_landmarks(landmarks, indices_necesarios):
            raise ValueError("Landmarks inválidos o incompletos.")

        # Evaluar ambas piernas
        resultados = []
        mensajes = []

        for lado, (cadera_idx, rodilla_idx, tobillo_idx) in {
            "izquierda": (
                PoseLandmark.LEFT_HIP.value,
                PoseLandmark.LEFT_KNEE.value,
                PoseLandmark.LEFT_ANKLE.value,
            ),
            "derecha": (
                PoseLandmark.RIGHT_HIP.value,
                PoseLandmark.RIGHT_KNEE.value,
                PoseLandmark.RIGHT_ANKLE.value,
            ),
        }.items():
            cadera = landmarks[cadera_idx]
            rodilla = landmarks[rodilla_idx]
            tobillo = landmarks[tobillo_idx]

            angulo_rodilla, sentadilla_valida = evaluar_sentadilla_pierna(
                cadera, rodilla, tobillo, angulo_min, angulo_max
            )
            resultados.append((angulo_rodilla, sentadilla_valida))
            mensajes.append(
                f"Pierna {lado}: Ángulo de la rodilla: {angulo_rodilla:.2f}°, "
                f"Sentadilla {'válida' if sentadilla_valida else 'no válida'}"
            )

        return {"mensajes": mensajes, "datos": resultados}

    except Exception as e:
        print(f"Error en evaluar_sentadillas: {e}")
        return {"mensajes": ["Error en la evaluación de la sentadilla"], "datos": []}