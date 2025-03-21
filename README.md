# DballTrainer

## Resumen del Proyecto

DballTrainer es una aplicación diseñada para analizar y evaluar la postura de jugadores de voleibol mediante el procesamiento de videos. Utiliza técnicas de visión por computadora para identificar movimientos clave y generar métricas sobre la calidad técnica de cada jugada.

### Funcionalidades principales:

1. **Conversión de Videos MOV a MP4:**

   - Convertimos todos los archivos .MOV en la carpeta `Videos` a archivos .MP4 utilizando `moviepy`.

2. **Detección y Evaluación de Posturas:**

   - Evaluamos diferentes aspectos de la postura del jugador (posición, movimiento, contacto, seguimiento, estabilidad).
   - Integramos estas evaluaciones en `main.py` para procesar videos y generar análisis de postura.

3. **Generación de Archivos de Salida:**

   - Guardamos los videos de salida y los archivos CSV de análisis de postura en la carpeta `salidas`.
   - Aseguramos que los nombres de los archivos de salida sean únicos para evitar sobrescribir archivos existentes.

4. **Pendiente:**

   - Hacer que el video también se reciba de fuentes online como YouTube.

## Requisitos

Asegúrate de tener los siguientes requisitos antes de ejecutar la aplicación:

- Python 3.x
- Las dependencias listadas en `requirements.txt`

## Instalación

Sigue estos pasos para instalar y configurar el proyecto:

1. Clona el repositorio:

2. Crea un entorno virtual:

3. Activa el entorno virtual:

   - En Windows:
   - En macOS/Linux:

4. Instala las dependencias:

## Uso

Para ejecutar la aplicación, usa el siguiente comando:

## Estructura del Proyecto

- `main.py` : Archivo principal para ejecutar la aplicación.
- `requirements.txt` : Lista de dependencias necesarias para el proyecto.
- `detecciones/` : Carpeta que contiene [descripción del contenido].
- `evaluaciones/` : Carpeta que contiene [descripción del contenido].
- `fonts/` : Carpeta que contiene [descripción del contenido].

##