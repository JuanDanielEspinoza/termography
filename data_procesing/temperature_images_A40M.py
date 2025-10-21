import cv2
import numpy as np
import os
from pathlib import Path
import subprocess
import json

# Configuración
carpeta_raw = r"C:\Users\ASUS\Desktop\Canada\imagenes_raw"
carpeta_salida = r"C:\Users\ASUS\Desktop\Canada\imagenes_temperatura"
carpeta_original = r"C:\Users\ASUS\Desktop\Canada\imagenes_radiometricas"

# Crear carpeta de salida si no existe
Path(carpeta_salida).mkdir(parents=True, exist_ok=True)

archivos_procesados = 0
archivos_error = 0

# Procesar cada imagen RAW
for archivo in os.listdir(carpeta_raw):
    if not archivo.lower().endswith('.png'):
        continue

    # Obtener el nombre del archivo original (sin "_raw")
    nombre_base = archivo.replace("_raw.png", "")
    archivo_original = nombre_base + ".jpg"
    
    ruta_raw = os.path.join(carpeta_raw, archivo)
    ruta_original = os.path.join(carpeta_original, archivo_original)
    nombre_salida = nombre_base + "_temp.png"
    ruta_salida = os.path.join(carpeta_salida, nombre_salida)

    print(f"🌡️ Procesando: {archivo}")

    try:
        # Extraer metadatos de calibración con exiftool
        result = subprocess.run(
            ["exiftool", "-j", "-Emissivity", "-PlanckR1", "-PlanckR2", 
             "-PlanckB", "-PlanckF", "-PlanckO", ruta_original],
            capture_output=True,
            text=True,
            check=True
        )
        
        metadata = json.loads(result.stdout)[0]
        
        # Parámetros de calibración
        emissivity = float(metadata.get('Emissivity', 0.96))
        R1 = float(metadata.get('PlanckR1', 19839.34))
        R2 = float(metadata.get('PlanckR2', 0.007745727))
        B = float(metadata.get('PlanckB', 1482.6))
        F = float(metadata.get('PlanckF', 1.1))
        O = float(metadata.get('PlanckO', -4096))
        
        print(f"   📊 Parámetros: E={emissivity}, R1={R1:.2f}, B={B:.2f}")
        
        # Cargar imagen RAW térmica
        raw_img = cv2.imread(ruta_raw, cv2.IMREAD_ANYDEPTH)
        
        if raw_img is None:
            print(f"   ❌ No se pudo cargar la imagen RAW")
            archivos_error += 1
            continue
        
        # Convertir a float32 para cálculos
        raw_img = raw_img.astype(np.float32)
        
        # Corrección por emisividad
        raw_corrected = raw_img / emissivity
        
        # Conversión a temperatura en Kelvin
        temperature_kelvin = B / np.log(R1 / (R2 * (raw_corrected + O)) + F)
        
        # Conversión a Celsius
        temperature_celsius = temperature_kelvin - 273.15
        
        # Mostrar estadísticas
        temp_min = np.min(temperature_celsius)
        temp_max = np.max(temperature_celsius)
        temp_mean = np.mean(temperature_celsius)
        
        print(f"   🌡️  Temp. Min: {temp_min:.2f}°C | Max: {temp_max:.2f}°C | Media: {temp_mean:.2f}°C")
          # Normalizar para visualización (mapear temperatura a 0-255)
        temp_normalized = ((temperature_celsius - temp_min) / (temp_max - temp_min) * 255).astype(np.uint8)
        
        # Aplicar paleta de colores INFERNO (similar a Iron de FLIR)
        temp_colored = cv2.applyColorMap(temp_normalized, cv2.COLORMAP_INFERNO)
        
        # Redimensionar a 160x120 (tamaño del sensor térmico)
        temp_resized = cv2.resize(temp_colored, (160, 120), interpolation=cv2.INTER_AREA)
        print(f"   📐 Redimensionado a: 160x120")
        
        # Rotar 90 grados en sentido horario
        temp_rotated = cv2.rotate(temp_resized, cv2.ROTATE_90_CLOCKWISE)
        print(f"   🔄 Rotado 90° en sentido horario")
        
        # Guardar imagen con paleta de colores
        cv2.imwrite(ruta_salida, temp_rotated)
        
        # También guardar los datos de temperatura como archivo numpy (opcional)
        ruta_npy = os.path.join(carpeta_salida, nombre_base + "_temp.npy")
        np.save(ruta_npy, temperature_celsius)
        
        print(f"   ✅ Guardado como: {nombre_salida}")
        archivos_procesados += 1

    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error al extraer metadatos: {e.stderr}")
        archivos_error += 1
    except KeyError as e:
        print(f"   ❌ Metadato faltante: {e}")
        archivos_error += 1
    except Exception as e:
        print(f"   ❌ Error inesperado: {str(e)}")
        archivos_error += 1

print(f"\n🎉 Proceso completado:")
print(f"   ✅ {archivos_procesados} imágenes procesadas")
if archivos_error > 0:
    print(f"   ❌ {archivos_error} imágenes con errores")
print(f"   📁 Imágenes guardadas en: {carpeta_salida}")
print(f"   💾 Archivos .npy con datos de temperatura también guardados")