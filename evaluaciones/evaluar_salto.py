import time
import mediapipe as mp
from typing import List, Dict, Union

mp_pose = mp.solutions.pose

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

def obtener_altura_pie(landmarks: List) -> Union[float, None]:
    """
    Obtiene la altura del pie más bajo entre el izquierdo y el derecho.
    Args:
        landmarks (list): Lista de landmarks detectados por MediaPipe.
    Returns:
        float: Altura del pie más bajo o None si ocurre un error.
    """
    try:
        altura_pie_izq = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y
        altura_pie_der = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y
        return min(altura_pie_izq, altura_pie_der)  # Tomar el pie más bajo
    except Exception as e:
        print(f"Error al obtener la altura del pie: {e}")
        return None

def evaluar_salto(
    landmarks: List,
    tiempo_inicio: float,
    tiempo_fin: float,
    altura_inicial: float = None
) -> Dict[str, Union[str, float]]:
    """
    Evalúa la altura y el tiempo de vuelo del salto.
    Args:
        landmarks (list): Lista de landmarks detectados por MediaPipe.
        tiempo_inicio (float): Tiempo en el que el salto comenzó.
        tiempo_fin (float): Tiempo en el que el salto terminó.
        altura_inicial (float, opcional): Altura inicial del pie para calcular la altura relativa.
    Returns:
        dict: Resultado de la evaluación con altura, tiempo de vuelo y mensaje descriptivo.
    """
    try:
        # Validar tiempos
        if tiempo_inicio >= tiempo_fin:
            raise ValueError("El tiempo de inicio debe ser menor que el tiempo de fin.")

        # Validar landmarks
        indices_necesarios = [
            mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
            mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
        ]
        if not validar_landmarks(landmarks, indices_necesarios):
            raise ValueError("Landmarks inválidos o incompletos.")

        # Obtener la altura del pie más bajo
        altura_pie = obtener_altura_pie(landmarks)
        if altura_pie is None:
            raise ValueError("No se pudo calcular la altura del pie.")

        # Calcular altura relativa si se proporciona altura inicial
        if altura_inicial is not None:
            altura_salto = max(0, altura_inicial - altura_pie)  # Altura relativa
        else:
            altura_salto = altura_pie  # Usar la altura absoluta si no hay referencia

        # Calcular tiempo de vuelo
        tiempo_vuelo = tiempo_fin - tiempo_inicio

        # Formatear resultados
        mensaje = (
            f"Altura del salto: {altura_salto:.2f}, "
            f"Tiempo de vuelo: {tiempo_vuelo:.2f}s"
        )
        return {"mensaje": mensaje, "altura": altura_salto, "tiempo_vuelo": tiempo_vuelo}

    except Exception as e:
        print(f"Error en evaluar_salto: {e}")
        return {"mensaje": "Error al evaluar el salto", "altura": 0, "tiempo_vuelo": 0}