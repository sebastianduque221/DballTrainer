def normalizar_landmarks(landmarks, altura, ancho):
    """
    Normaliza los landmarks en relación con el tamaño de la imagen.
    Args:
        landmarks (list): Lista de landmarks detectados por MediaPipe.
        altura (int): Altura de la imagen.
        ancho (int): Ancho de la imagen.
    Returns:
        list: Lista de landmarks normalizados.
    """
    return [
        {"x": lm.x * ancho, "y": lm.y * altura, "z": lm.z}
        for lm in landmarks
    ]