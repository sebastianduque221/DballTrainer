import math
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from mediapipe.python.solutions.pose import PoseLandmark

class GeometryCalculator:
    """Clase responsable de cálculos geométricos básicos."""
    
    @staticmethod
    def calculate_angle(p1: PoseLandmark, p2: PoseLandmark, p3: PoseLandmark) -> Optional[float]:
        """Calcula el ángulo entre tres puntos."""
        try:
            angle = math.degrees(
                math.atan2(p3.y - p2.y, p3.x - p2.x) -
                math.atan2(p1.y - p2.y, p1.x - p2.x)
            )
            return abs(angle) if angle >= 0 else abs(angle + 360)
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return None
    
    @staticmethod
    def calculate_distance(p1: PoseLandmark, p2: PoseLandmark) -> Optional[float]:
        """Calcula la distancia euclidiana entre dos puntos."""
        try:
            return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return None

class LandmarkValidator:
    """Clase responsable de validar los landmarks."""
    
    REQUIRED_LANDMARKS = [
        PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST,
        PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST,
        PoseLandmark.NOSE, PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP
    ]
    
    @classmethod
    def validate_landmarks(cls, landmarks: List) -> bool:
        """Valida que los landmarks requeridos estén presentes."""
        if not isinstance(landmarks, list):
            return False
        return all(landmark.value < len(landmarks) for landmark in cls.REQUIRED_LANDMARKS)

class BlockEvaluator(ABC):
    """Interfaz abstracta para evaluadores de bloqueo."""
    
    @abstractmethod
    def evaluate(self, landmarks: List) -> Dict:
        pass
    
    @abstractmethod
    def get_headers(self) -> List[str]:
        pass

class VolleyballBlockEvaluator(BlockEvaluator):
    """Evaluador específico para la técnica de bloqueo en voleibol."""
    
    def __init__(self, symmetry_tolerance: int = 15):
        self._geometry_calculator = GeometryCalculator()
        self._symmetry_tolerance = symmetry_tolerance
    
    def evaluate(self, landmarks: List) -> Dict:
        """Evalúa la técnica de bloqueo."""
        try:
            if not LandmarkValidator.validate_landmarks(landmarks):
                raise ValueError("Missing essential landmarks for block evaluation.")
            
            # Extraer landmarks relevantes
            extracted = self._extract_landmarks(landmarks)
            
            # Calcular métricas
            metrics = self._calculate_metrics(extracted)
            
            # Evaluar criterios de bloqueo
            evaluation = self._evaluate_block(metrics)
            
            # Generar mensajes
            messages = self._generate_messages(metrics, evaluation)
            
            return {
                "messages": messages,
                "data": [
                    metrics["left_arm_angle"],
                    metrics["right_arm_angle"],
                    metrics["left_block_height"],
                    metrics["right_block_height"],
                    metrics["torso_alignment"],
                    evaluation["is_valid_block"],
                    metrics["hands_separation"],
                    evaluation["is_symmetrical"]
                ]
            }
            
        except Exception as e:
            print(f"Error in block evaluation: {e}")
            return {
                "messages": ["Error in block detection"],
                "data": [None] * 8
            }
    
    def get_headers(self) -> List[str]:
        """Devuelve los encabezados para la salida."""
        return [
            "Angulo Brazo Izq", "Angulo Brazo Der", "Altura Bloqueo Izq", "Altura Bloqueo Der",
            "Alineación Tronco", "Bloqueo Válido", "Separación de Manos", "Simetría"
        ]
    
    def _extract_landmarks(self, landmarks: List) -> Dict:
        """Extrae los landmarks relevantes."""
        return {
            "left_shoulder": landmarks[PoseLandmark.LEFT_SHOULDER.value],
            "left_elbow": landmarks[PoseLandmark.LEFT_ELBOW.value],
            "left_wrist": landmarks[PoseLandmark.LEFT_WRIST.value],
            "right_shoulder": landmarks[PoseLandmark.RIGHT_SHOULDER.value],
            "right_elbow": landmarks[PoseLandmark.RIGHT_ELBOW.value],
            "right_wrist": landmarks[PoseLandmark.RIGHT_WRIST.value],
            "head": landmarks[PoseLandmark.NOSE.value],
            "left_hip": landmarks[PoseLandmark.LEFT_HIP.value],
            "right_hip": landmarks[PoseLandmark.RIGHT_HIP.value]
        }
    
    def _calculate_metrics(self, landmarks: Dict) -> Dict:
        """Calcula todas las métricas necesarias."""
        return {
            "left_arm_angle": self._geometry_calculator.calculate_angle(
                landmarks["left_shoulder"], landmarks["left_elbow"], landmarks["left_wrist"]),
            "right_arm_angle": self._geometry_calculator.calculate_angle(
                landmarks["right_shoulder"], landmarks["right_elbow"], landmarks["right_wrist"]),
            "left_block_height": landmarks["left_wrist"].y / landmarks["head"].y,
            "right_block_height": landmarks["right_wrist"].y / landmarks["head"].y,
            "torso_alignment": self._geometry_calculator.calculate_angle(
                landmarks["left_hip"], landmarks["left_shoulder"], landmarks["right_shoulder"]),
            "hands_separation": self._geometry_calculator.calculate_distance(
                landmarks["left_wrist"], landmarks["right_wrist"])
        }
    
    def _evaluate_block(self, metrics: Dict) -> Dict:
        """Evalúa los criterios de un bloqueo válido."""
        left_arm_valid = metrics["left_arm_angle"] is not None and metrics["left_arm_angle"] > 160
        right_arm_valid = metrics["right_arm_angle"] is not None and metrics["right_arm_angle"] > 160
        left_height_valid = metrics["left_block_height"] is not None and metrics["left_block_height"] < 0.5
        right_height_valid = metrics["right_block_height"] is not None and metrics["right_block_height"] < 0.5
        torso_aligned = (metrics["torso_alignment"] is not None and 
                        75 <= metrics["torso_alignment"] <= 105)
        
        symmetrical = (metrics["left_arm_angle"] is not None and 
                      metrics["right_arm_angle"] is not None and
                      abs(metrics["left_arm_angle"] - metrics["right_arm_angle"]) < self._symmetry_tolerance)
        
        return {
            "is_valid_block": all([
                left_arm_valid, right_arm_valid, 
                left_height_valid, right_height_valid,
                torso_aligned
            ]),
            "is_symmetrical": symmetrical,
            "left_arm_valid": left_arm_valid,
            "right_arm_valid": right_arm_valid,
            "left_height_valid": left_height_valid,
            "right_height_valid": right_height_valid,
            "torso_aligned": torso_aligned
        }
    
    def _generate_messages(self, metrics: Dict, evaluation: Dict) -> List[str]:
        """Genera mensajes descriptivos basados en los resultados."""
        return [
            f"Ángulo brazo izquierdo: {metrics['left_arm_angle']:.2f}° "
            f"({'Correcto' if evaluation['left_arm_valid'] else 'Incorrecto'})",
            f"Ángulo brazo derecho: {metrics['right_arm_angle']:.2f}° "
            f"({'Correcto' if evaluation['right_arm_valid'] else 'Incorrecto'})",
            f"Altura muñeca izquierda: "
            f"{'Correcta' if evaluation['left_height_valid'] else 'Demasiado baja'}",
            f"Altura muñeca derecha: "
            f"{'Correcta' if evaluation['right_height_valid'] else 'Demasiado baja'}",
            f"Tronco {'Alineado' if evaluation['torso_aligned'] else 'Desalineado'}",
            f"Bloqueo {'Válido' if evaluation['is_valid_block'] else 'No válido'}",
            f"Separación de manos: {metrics['hands_separation']:.2f}",
            f"Simetría entre brazos: {'Correcta' if evaluation['is_symmetrical'] else 'Incorrecta'}"
        ]

class BloqueoEvaluator:
    def evaluate(self, landmarks):
        # Implementa aquí la lógica real de evaluación del bloqueo
        return {
            "mensajes": ["Evaluación de bloqueo no implementada."],
            "datos": [None] * 8  # Ajusta la cantidad de datos según tus encabezados
        }

def detectar_bloqueo(landmarks: List, tolerancia_simetria: int = 15) -> Dict:
    """Función de alto nivel para detectar la técnica de bloqueo."""
    evaluator = VolleyballBlockEvaluator(symmetry_tolerance=tolerancia_simetria)
    return evaluator.evaluate(landmarks)

def obtener_encabezados_bloqueo():
    return [
        "Angulo Brazo Izq", "Angulo Brazo Der", "Altura Bloqueo Izq", "Altura Bloqueo Der",
        "Alineación Tronco", "Bloqueo Válido", "Separación de Manos", "Simetría"
    ]