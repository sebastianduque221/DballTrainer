FROM python:3.11-slim

# Evita buffers raros en logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependencias del sistema (MediaPipe, OpenCV)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiamos solo requirements primero (capa cacheable)
COPY requirements.txt .

# Instalamos dependencias Python
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código
COPY . .

# Puerto FastAPI
EXPOSE 8000

# Comando de arranque
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
