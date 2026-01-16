import cv2

def dibujar_texto(image, texto, posicion, color=(255, 0, 0), tamaño=1):
    """
    Dibuja texto en una imagen.
    Args:
        image (numpy.ndarray): Imagen en la que se dibujará el texto.
        texto (str): Texto a dibujar.
        posicion (tuple): Coordenadas (x, y) para el texto.
        color (tuple): Color del texto en formato BGR.
        tamaño (int): Tamaño de la fuente.
    """
    cv2.putText(image, texto, posicion, cv2.FONT_HERSHEY_SIMPLEX, tamaño, color, 2)