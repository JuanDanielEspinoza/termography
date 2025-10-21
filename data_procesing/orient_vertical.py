import cv2
import os
from pathlib import Path

# Configuraciónv
carpeta_entrada = r"C:\Users\ASUS\Desktop\Canada\procesar_flir_one_edge"
carpeta_salida = r"C:\Users\ASUS\Desktop\Canada\imagenes_vertical_flir_one"

# Crear carpeta de salida si no existe
Path(carpeta_salida).mkdir(parents=True, exist_ok=True)

archivos_procesados = 0
archivos_error = 0

# Extensiones de imagen soportadas
extensiones_imagen = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

print("🔄 Procesando imágenes para orientación vertical...\n")

for archivo in os.listdir(carpeta_entrada):
    extension = os.path.splitext(archivo)[1].lower()
    
    if extension not in extensiones_imagen:
        continue
    
    ruta_entrada = os.path.join(carpeta_entrada, archivo)
    ruta_salida = os.path.join(carpeta_salida, archivo)
    
    print(f"📷 Procesando: {archivo}")
    
    try:
        # Cargar la imagen
        imagen = cv2.imread(ruta_entrada)
        
        if imagen is None:
            print(f"   ❌ No se pudo cargar la imagen")
            archivos_error += 1
            continue
        
        altura, ancho = imagen.shape[:2]
        print(f"   Tamaño original: {ancho}x{altura}")
        
        # Si la imagen es horizontal (ancho > altura), rotarla 90° en sentido horario
        if ancho > altura:
            imagen = cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE)
            print(f"   🔄 Rotada 90° (era horizontal)")
            altura, ancho = imagen.shape[:2]
        else:
            print(f"   ✓ Ya está vertical")
        
        # Redimensionar a 120x160 (ancho x alto)
        imagen_resized = cv2.resize(imagen, (120, 160), interpolation=cv2.INTER_AREA)
        print(f"   📐 Redimensionada a: 120x160")
        
        # Guardar la imagen procesada
        cv2.imwrite(ruta_salida, imagen_resized)
        
        print(f"   ✅ Guardada como: {archivo}\n")
        archivos_procesados += 1
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}\n")
        archivos_error += 1

print(f"🎉 Proceso completado:")
print(f"   ✅ {archivos_procesados} imágenes procesadas")
if archivos_error > 0:
    print(f"   ❌ {archivos_error} imágenes con errores")
print(f"   📁 Imágenes guardadas en: {carpeta_salida}")
