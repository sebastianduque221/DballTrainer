"""
Paquete de utilidades para el procesamiento del proyecto.

Se exponen:
  - Configuración global desde config.py (por ejemplo, CSV_HEADERS).
  - Funciones y constantes relacionadas con la generación de encabezados CSV desde CSVheader.py.
  - Utilidades para operaciones con poses desde pose_utils.py.
  - Funciones de procesamiento de frames (por ejemplo, procesar_frame) desde procesamiento.py.
  - Funciones de manejo de video desde video_utils.py.
"""

from .config import CSV_HEADERS
from .pose_utils import *              # Se importan todas las funciones de ayuda sobre poses
from .procesamiento import procesar_frame
from .video_utils import *             # Se importan todas las utilidades relacionadas con video

__all__ = [
    "CSV_HEADERS",
    "get_csv_headers",
    "procesar_frame",
    # Los nombres exportados por pose_utils y video_utils quedan disponibles vía import *
]
