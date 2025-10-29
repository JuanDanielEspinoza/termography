import cv2
import os
import numpy as np
from pathlib import Path

# Configuración
carpeta_entrada = r"C:\Users\ASUS\Desktop\Canada_Repository\termography\data\data_roboflow_flir_one_160_120"
carpeta_salida = r"C:\Users\ASUS\Desktop\Canada_Repository\termography\data\processed_flir_images_no_logo"

# Crear carpeta de salida si no existe
Path(carpeta_salida).mkdir(parents=True, exist_ok=True)

def process_flir_image(image_path, output_path):
    """
    Procesar imagen FLIR: redimensionar y eliminar logo
    """
    # Leer imagen
    frame = cv2.imread(image_path)
    if frame is None:
        return False
    
    print(f"   📐 Tamaño original: {frame.shape}")
    
    # Redimensionar al tamaño del sensor térmico original (160x120)
    # Nota: OpenCV usa (ancho, alto), pero queremos 160x120 térmico
    frame_resized = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
    
    # Eliminar logo de FLIR usando inpainting
    # Crear una máscara del logo (ajusta las coordenadas según la posición del logo)
    mask = np.zeros(frame_resized.shape[:2], dtype=np.uint8)
    mask[0:15, 0:35] = 255  # Área del logo en la esquina superior izquierda
    
    # Usar inpainting para rellenar la región del logo
    frame_processed = cv2.inpaint(frame_resized, mask, 3, cv2.INPAINT_TELEA)

    imagen_rotada = cv2.rotate(frame_processed, cv2.ROTATE_90_CLOCKWISE)
    
    # Guardar imagen procesada
    cv2.imwrite(output_path, imagen_rotada)
    
    return True

# Extensiones de imagen soportadas
extensiones_imagen = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

print("🖼️ PROCESAMIENTO DE IMÁGENES FLIR")
print(f"📁 Carpeta entrada: {carpeta_entrada}")
print(f"📁 Carpeta salida: {carpeta_salida}")
print("=" * 60)

# Verificar que la carpeta de entrada existe
if not os.path.exists(carpeta_entrada):
    print(f"❌ Error: La carpeta de entrada no existe: {carpeta_entrada}")
    exit(1)

# Obtener lista de imágenes
archivos_imagen = []
for archivo in os.listdir(carpeta_entrada):
    extension = os.path.splitext(archivo)[1].lower()
    if extension in extensiones_imagen:
        archivos_imagen.append(archivo)

if not archivos_imagen:
    print("❌ No se encontraron imágenes en la carpeta especificada")
    exit(1)

print(f"📊 Encontradas {len(archivos_imagen)} imágenes")

# Procesar cada imagen
procesadas = 0
errores = 0

for i, archivo in enumerate(archivos_imagen, 1):
    print(f"\n📷 [{i}/{len(archivos_imagen)}] Procesando: {archivo}")
    
    ruta_entrada = os.path.join(carpeta_entrada, archivo)
    
    # Crear nombre de salida con sufijo "_processed"
    nombre_base = os.path.splitext(archivo)[0]
    extension = os.path.splitext(archivo)[1]
    nombre_salida = f"{nombre_base}_flir_processed{extension}"
    ruta_salida = os.path.join(carpeta_salida, nombre_salida)
    
    try:
        if process_flir_image(ruta_entrada, ruta_salida):
            print(f"   ✅ Procesada exitosamente → {nombre_salida}")
            procesadas += 1
        else:
            print(f"   ❌ Error al leer la imagen")
            errores += 1
    except Exception as e:
        print(f"   ❌ Error procesando: {str(e)}")
        errores += 1

print(f"\n{'='*60}")
print(f"🎉 PROCESAMIENTO COMPLETADO")
print(f"   ✅ Imágenes procesadas: {procesadas}")
print(f"   ❌ Errores: {errores}")
print(f"   📁 Imágenes guardadas en: {carpeta_salida}")

if procesadas > 0:
    print(f"\n📋 CAMBIOS APLICADOS:")
    print(f"   • 📐 Redimensionado a 160x120 (resolución FLIR One)")
    print(f"   • 🚫 Logo FLIR eliminado (esquina superior izquierda)")
    print(f"   • 🎨 Inpainting aplicado para rellenar área del logo")
    
    print(f"\n💡 AJUSTES DISPONIBLES:")
    print(f"   Si el logo no se elimina correctamente, ajusta las coordenadas:")
    print(f"   mask[y1:y2, x1:x2] = 255  # Donde (x1,y1) y (x2,y2) son las esquinas del logo")
    print(f"   Coordenadas actuales: mask[0:15, 0:35] = 255")
    
    # Mostrar información de una imagen procesada
    if archivos_imagen:
        sample_file = os.path.join(carpeta_salida, f"{os.path.splitext(archivos_imagen[0])[0]}_flir_processed{os.path.splitext(archivos_imagen[0])[1]}")
        if os.path.exists(sample_file):
            sample_img = cv2.imread(sample_file)
            if sample_img is not None:
                print(f"\n📊 Tamaño final de imágenes: {sample_img.shape} (H x W x C)")

print(f"\n🏁 Proceso terminado")