import cv2
import numpy as np
import os
from pathlib import Path
import subprocess
import json

# 📂 Configuración
carpeta_raw = r"C:\Users\ASUS\Desktop\Canada\imagenes_raw_flir_one"
carpeta_salida = r"C:\Users\ASUS\Desktop\Canada\imagenes_temperatura_flir_one"

carpeta_original = r"C:\Users\ASUS\Desktop\Canada\procesar_flir_one_edge"   # donde están los .jpg originales

# Crear carpeta de salida si no existe
Path(carpeta_salida).mkdir(parents=True, exist_ok=True)

archivos_procesados = 0
archivos_error = 0

# Procesar cada imagen RAW
for archivo in os.listdir(carpeta_raw):
    if not archivo.lower().endswith('.png'):
        continue

    nombre_base = archivo.replace("_raw.png", "")
    archivo_original = nombre_base + ".jpg"
    
    ruta_raw = os.path.join(carpeta_raw, archivo)
    ruta_original = os.path.join(carpeta_original, archivo_original)
    nombre_salida = nombre_base + "_temp.png"
    ruta_salida = os.path.join(carpeta_salida, nombre_salida)

    print(f"🌡️ Procesando: {archivo}")

    try:
        # 📥 Extraer metadatos relevantes con exiftool
        result = subprocess.run(
            [
                "exiftool", "-j",
                "-Emissivity", "-PlanckR1", "-PlanckR2",
                "-PlanckB", "-PlanckF", "-PlanckO",
                ruta_original
            ],
            capture_output=True,
            text=True,
            check=True
        )
        metadata = json.loads(result.stdout)[0]

        # 🧪 Parámetros de calibración
        emissivity = float(metadata.get('Emissivity', 0.95))
        R1 = float(metadata.get('PlanckR1', 16201.165))
        R2 = float(metadata.get('PlanckR2', 0.018284522))
        B  = float(metadata.get('PlanckB', 1421.5))
        F  = float(metadata.get('PlanckF', 1.0))
        O  = float(metadata.get('PlanckO', -1381))

        print(f"   📊 Parámetros: E={emissivity}, R1={R1:.2f}, R2={R2:.6f}, B={B:.2f}, F={F:.2f}, O={O}")

        # 🖼️ Cargar imagen RAW térmica (16 bits)
        raw_img = cv2.imread(ruta_raw, cv2.IMREAD_ANYDEPTH)
        if raw_img is None:
            print(f"   ❌ No se pudo cargar la imagen RAW: {ruta_raw}")
            archivos_error += 1
            continue

        raw_img = raw_img.astype(np.float32)

        # 🌡️ Conversión a temperatura (según fórmula de FLIR)
        raw_corrected = raw_img / emissivity
        temperature_kelvin = B / np.log(R1 / (R2 * (raw_corrected + O)) + F)
        temperature_celsius = temperature_kelvin - 273.15

        # 📈 Estadísticas
        temp_min = float(np.min(temperature_celsius))
        temp_max = float(np.max(temperature_celsius))
        temp_mean = float(np.mean(temperature_celsius))
        print(f"   🌡️ Min: {temp_min:.2f}°C | Max: {temp_max:.2f}°C | Media: {temp_mean:.2f}°C")

        # 🌈 Normalizar para visualización
        temp_norm = ((temperature_celsius - temp_min) / (temp_max - temp_min) * 255).astype(np.uint8)
        temp_colored = cv2.applyColorMap(temp_norm, cv2.COLORMAP_INFERNO)

        # 💾 Guardar imagen visual
        cv2.imwrite(ruta_salida, temp_colored)

        # 💾 Guardar datos en numpy (para análisis cuantitativo)
        ruta_npy = os.path.join(carpeta_salida, nombre_base + "_temp.npy")
        np.save(ruta_npy, temperature_celsius)

        print(f"   ✅ Guardado: {nombre_salida}")
        archivos_procesados += 1

    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error al extraer metadatos con exiftool: {e.stderr}")
        archivos_error += 1
    except KeyError as e:
        print(f"   ❌ Metadato faltante: {e}")
        archivos_error += 1
    except Exception as e:
        print(f"   ❌ Error inesperado: {str(e)}")
        archivos_error += 1

# 📊 Resumen final
print(f"\n🎉 Proceso completado:")
print(f"   ✅ {archivos_procesados} imágenes procesadas correctamente")
if archivos_error > 0:
    print(f"   ❌ {archivos_error} imágenes con errores")
print(f"   📁 Temperaturas guardadas en: {carpeta_salida}")
print(f"   💾 Archivos .npy listos para análisis numérico")
