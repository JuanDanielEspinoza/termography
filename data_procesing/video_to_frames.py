import cv2
import os
import numpy as np
from pathlib import Path

# Configuración
carpeta_videos = r"C:\Users\ASUS\Desktop\Canada\test_uv\videos"
carpeta_salida = r"C:\Users\ASUS\Desktop\Canada\test_uv\sujeto05_imagenes_termicas"


# Crear carpeta de salida si no existe
Path(carpeta_salida).mkdir(parents=True, exist_ok=True)

# Extensiones de video soportadas
extensiones_video = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

# Procesar cada video en la carpeta
for archivo in os.listdir(carpeta_videos):
    ruta_video = os.path.join(carpeta_videos, archivo)
    
    # Verificar si es un archivo de video
    if not os.path.isfile(ruta_video):
        continue
    
    extension = os.path.splitext(archivo)[1].lower()
    if extension not in extensiones_video:
        continue
    
    print(f"\n📹 Procesando: {archivo}")
    
    # Abrir video
    cap = cv2.VideoCapture(ruta_video)
    
    if not cap.isOpened():
        print(f"❌ Error al abrir el video: {archivo}")
        continue
    
    # Obtener información del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   FPS: {fps:.2f}")
    print(f"   Total de frames: {total_frames}")
    print(f"   Se extraerán 9 frames del video")
    
    # Nombre base del video (sin extensión)
    nombre_base = os.path.splitext(archivo)[0]
    
    # Calcular qué frames extraer (distribuidos uniformemente)
    frames_a_extraer = 9
    intervalo = total_frames // (frames_a_extraer + 1)
    frames_objetivo = [intervalo * (i + 1) for i in range(frames_a_extraer)]
    
    # Extraer frames
    frame_num = 0
    frames_guardados = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Solo procesar los frames seleccionados
        if frame_num not in frames_objetivo:
            frame_num += 1
            continue
        
        # Redimensionar al tamaño del sensor térmico original (160x120)
        frame_resized = cv2.resize(frame, (120, 160), interpolation=cv2.INTER_AREA)
        
        # Eliminar logo de FLIR usando inpainting
        # Crear una máscara del logo (ajusta las coordenadas según la posición del logo)
        mask = np.zeros(frame_resized.shape[:2], dtype=np.uint8)
        mask[0:15, 0:35] = 255  # Área del logo en la esquina superior izquierda
        
        # Usar inpainting para rellenar la región del logo
        frame_resized = cv2.inpaint(frame_resized, mask, 3, cv2.INPAINT_TELEA)
        
        # Rotar 90 grados hacia la derecha (sentido horario)
        #frame_rotated = cv2.rotate(frame_resized, cv2.ROTATE_90_CLOCKWISE)
        
        # Guardar frame redimensionado y rotado
        segundo = frames_guardados
        nombre_frame = f"{nombre_base}_sec_{segundo:04d}.jpg"
        ruta_salida = os.path.join(carpeta_salida, nombre_frame)
        cv2.imwrite(ruta_salida, frame_resized)
        
        frames_guardados += 1
        frame_num += 1
        
        # Mostrar progreso
        print(f"   Procesados: {frames_guardados}/2 frames", end='\r')
    
    cap.release()
    print(f"\n   ✅ {frames_guardados} frames guardados")

print(f"\n🎉 Proceso completado. Frames guardados en: {carpeta_salida}")
