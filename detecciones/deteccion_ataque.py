import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from mediapipe.python.solutions.pose import PoseLandmark

# ================ Domain Models ================
class EvaluationType(Enum):
    CONTACTO = "Contacto"
    ESTABILIDAD = "Estabilidad"
    MOVIMIENTO = "Movimiento"
    POSICION = "Posicion"
    SALTO = "Salto"
    SEGUIMIENTO = "Seguimiento"
    SENTADILLA = "Sentadilla"

@dataclass
class LandmarkData:
    x: float
    y: float
    visibility: float

@dataclass
class FrameData:
    frame_number: int
    landmarks: Dict[int, LandmarkData]
    timestamp: float

@dataclass
class AngleData:
    left_elbow: float
    right_elbow: float

@dataclass
class VelocityData:
    left_elbow: float
    right_elbow: float

@dataclass
class AttackResult:
    frame_number: int
    angles: AngleData
    velocities: VelocityData
    is_attack: bool
    is_valid_contact: bool
    is_symmetric: bool
    evaluation_scores: Dict[EvaluationType, float]
    message: str
    error: Optional[str] = None

@dataclass
class DetectionConfig:
    attack_angle_threshold: float = 60
    velocity_threshold: float = 300
    symmetry_tolerance: float = 25
    fps: float = 30
    noise_threshold: float = 2.0

# ================ Core Calculators ================
class AngleCalculator:
    @staticmethod
    def calculate_angle(p1: LandmarkData, p2: LandmarkData, p3: LandmarkData) -> Optional[float]:
        try:
            angle_rad = math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x)
            angle_deg = math.degrees(angle_rad)
            angle_deg = abs(angle_deg) % 360
            return min(angle_deg, 360 - angle_deg)
        except Exception as e:
            print(f"[ERROR] AngleCalculator: {str(e)}")
            return None

class VelocityCalculator:
    def __init__(self, noise_threshold: float = 2.0):
        self.noise_threshold = noise_threshold

    def calculate_velocity(self, current_angle: float, previous_angle: float, time_delta: float) -> float:
        if time_delta <= 0 or None in (current_angle, previous_angle):
            return 0.0

        current_angle = current_angle % 360
        previous_angle = previous_angle % 360
        
        difference = min(
            (current_angle - previous_angle) % 360,
            (previous_angle - current_angle) % 360
        )
        
        if difference <= self.noise_threshold:
            return 0.0
        
        return difference / time_delta

# ================ Evaluation System ================
class EvaluationContext:
    def __init__(self, landmarks: Dict[int, LandmarkData], velocities: VelocityData, previous_angles: Optional[AngleData]):
        self.landmarks = landmarks
        self.velocities = velocities
        self.previous_angles = previous_angles

class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, context: EvaluationContext) -> float:
        pass

class EvaluationFactory:
    def __init__(self):
        self._strategies = {}

    def register(self, eval_type: EvaluationType, strategy: EvaluationStrategy):
        self._strategies[eval_type] = strategy

    def get_evaluator(self, eval_type: EvaluationType) -> EvaluationStrategy:
        return self._strategies.get(eval_type)

    def evaluate_all(self, context: EvaluationContext) -> Dict[EvaluationType, float]:
        return {
            eval_type: strategy.evaluate(context)
            for eval_type, strategy in self._strategies.items()
        }

# ================ Concrete Evaluators ================
class ContactEvaluator(EvaluationStrategy):
    def evaluate(self, context: EvaluationContext) -> float:
        from evaluaciones.evaluar_contacto import evaluar_contacto
        return evaluar_contacto(context.landmarks)

class StabilityEvaluator(EvaluationStrategy):
    def evaluate(self, context: EvaluationContext) -> float:
        from evaluaciones.evaluar_estabilidad import evaluar_estabilidad
        return evaluar_estabilidad(context.landmarks)

class MovementEvaluator(EvaluationStrategy):
    def evaluate(self, context: EvaluationContext) -> float:
        from evaluaciones.evaluar_movimiento import evaluar_movimiento

        evaluador = evaluar_movimiento()  # o evaluar_movimiento(debug_mode=True)
        resultado = evaluador.analyze_volleyball_posture(context.landmarks)

        return resultado

class PositionEvaluator(EvaluationStrategy):
    def evaluate(self, context: EvaluationContext) -> float:
        from evaluaciones.evaluar_posicion import evaluar_posicion
        return evaluar_posicion(context.landmarks)

class JumpEvaluator(EvaluationStrategy):
    def evaluate(self, context: EvaluationContext) -> float:
        from evaluaciones.evaluar_salto import evaluar_salto
        return evaluar_salto(context.landmarks, context.previous_angles)

class TrackingEvaluator(EvaluationStrategy):
    def evaluate(self, context: EvaluationContext) -> float:
        from evaluaciones.evaluar_seguimiento import evaluar_seguimiento
        return evaluar_seguimiento(context.landmarks)

class SquatEvaluator(EvaluationStrategy):
    def evaluate(self, context: EvaluationContext) -> float:
        from evaluaciones.evaluar_sentadillas import evaluar_sentadillas
        return evaluar_sentadillas(context.landmarks)

# ================ Main Detection System ================
class detectar_ataque:
    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self._validate_initialization()
        self.angle_calculator = AngleCalculator()
        self.velocity_calculator = VelocityCalculator(self.config.noise_threshold)
        self.previous_data = None
        self.evaluation_factory = self._setup_evaluation_factory()

    def _validate_initialization(self):
        if not isinstance(self.config, DetectionConfig):
            raise TypeError("config debe ser una instancia de DetectionConfig")
        if not hasattr(self.config, 'noise_threshold'):
            raise AttributeError("DetectionConfig debe tener el atributo 'noise_threshold'")

    def _setup_evaluation_factory(self):
        factory = EvaluationFactory()
        factory.register(EvaluationType.CONTACTO, ContactEvaluator())
        factory.register(EvaluationType.ESTABILIDAD, StabilityEvaluator())
        factory.register(EvaluationType.MOVIMIENTO, MovementEvaluator())
        factory.register(EvaluationType.POSICION, PositionEvaluator())
        factory.register(EvaluationType.SALTO, JumpEvaluator())
        factory.register(EvaluationType.SEGUIMIENTO, TrackingEvaluator())
        factory.register(EvaluationType.SENTADILLA, SquatEvaluator())
        return factory

    def detect(self, frame_data: FrameData) -> AttackResult:
        try:
            if not isinstance(frame_data, FrameData):
                raise TypeError("frame_data must be a FrameData instance")
            if not frame_data.landmarks:
                raise ValueError("No landmarks provided")
            if not isinstance(frame_data.landmarks, dict):
                raise TypeError("Landmarks must be a dictionary")

            points = self._extract_landmarks(frame_data.landmarks)
            angles = self._calculate_angles(points)

            time_delta = 1 / self.config.fps if self.config.fps > 0 else 1 / 30

            velocities = VelocityData(0.0, 0.0)
            if self.previous_data and hasattr(self.previous_data, 'angles'):
                velocities = self._calculate_velocities(angles, time_delta)

            is_attack, is_valid_contact, is_symmetric = self._evaluate_attack(angles, velocities)

            eval_context = EvaluationContext(
                landmarks=frame_data.landmarks,
                velocities=velocities,
                previous_angles=self.previous_data.angles if self.previous_data else None
            )

            try:
                evaluation_scores = self.evaluation_factory.evaluate_all(eval_context)
            except Exception as eval_error:
                print(f"[WARNING] Evaluation error: {str(eval_error)}")
                evaluation_scores = {e: 0.0 for e in EvaluationType}

            message = (
                "Remate válido" if is_valid_contact else
                "Ataque detectado (sin contacto)" if is_attack else
                "Movimiento normal"
            )

            result = AttackResult(
                frame_number=frame_data.frame_number,
                angles=angles,
                velocities=velocities,
                is_attack=is_attack,
                is_valid_contact=is_valid_contact,
                is_symmetric=is_symmetric,
                evaluation_scores=evaluation_scores,
                message=message
            )

            self.previous_data = result
            return result

        except Exception as e:
            error_msg = f"Error processing frame {getattr(frame_data, 'frame_number', 'unknown')}: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return AttackResult(
                frame_number=getattr(frame_data, 'frame_number', -1),
                angles=AngleData(0, 0),
                velocities=VelocityData(0, 0),
                is_attack=False,
                is_valid_contact=False,
                is_symmetric=False,
                evaluation_scores={},
                message="Error en detección",
                error=error_msg
            )

    def _extract_landmarks(self, landmarks: Dict[int, LandmarkData]) -> Dict[str, LandmarkData]:
        return {
            "left_shoulder": landmarks.get(PoseLandmark.LEFT_SHOULDER.value),
            "left_elbow": landmarks.get(PoseLandmark.LEFT_ELBOW.value),
            "left_wrist": landmarks.get(PoseLandmark.LEFT_WRIST.value),
            "right_shoulder": landmarks.get(PoseLandmark.RIGHT_SHOULDER.value),
            "right_elbow": landmarks.get(PoseLandmark.RIGHT_ELBOW.value),
            "right_wrist": landmarks.get(PoseLandmark.RIGHT_WRIST.value),
        }

    def _calculate_angles(self, points: Dict[str, LandmarkData]) -> AngleData:
        left = self.angle_calculator.calculate_angle(
            points["left_shoulder"], points["left_elbow"], points["left_wrist"]
        )
        right = self.angle_calculator.calculate_angle(
            points["right_shoulder"], points["right_elbow"], points["right_wrist"]
        )
        return AngleData(left_elbow=left or 0.0, right_elbow=right or 0.0)

    def _calculate_velocities(self, angles: AngleData, time_delta: float) -> VelocityData:
        prev = self.previous_data.angles
        return VelocityData(
            left_elbow=self.velocity_calculator.calculate_velocity(angles.left_elbow, prev.left_elbow, time_delta),
            right_elbow=self.velocity_calculator.calculate_velocity(angles.right_elbow, prev.right_elbow, time_delta),
        )

    def _evaluate_attack(self, angles: AngleData, velocities: VelocityData) -> Tuple[bool, bool, bool]:
        is_attack = (
            angles.left_elbow < self.config.attack_angle_threshold or
            angles.right_elbow < self.config.attack_angle_threshold
        ) and (
            velocities.left_elbow > self.config.velocity_threshold or
            velocities.right_elbow > self.config.velocity_threshold
        )

        is_valid_contact = is_attack and (
            abs(velocities.left_elbow - velocities.right_elbow) < self.config.symmetry_tolerance
        )

        is_symmetric = abs(angles.left_elbow - angles.right_elbow) < self.config.symmetry_tolerance

        return is_attack, is_valid_contact, is_symmetric

# ================ Encabezados CSV ================
def obtener_encabezados_ataque() -> list:
    return [
        "Angulo Codo Izq", "Angulo Codo Der",
        "Velocidad Angular Codo Izq", "Velocidad Angular Codo Der",
        "Ataque Valido", "Contacto Valido", "Simetria",
        "Contacto", "Estabilidad", "Movimiento", "Posicion", "Salto", "Seguimiento", "Sentadilla"
    ]
