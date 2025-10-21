import os
import subprocess
import cv2
import numpy as np

# --- Configuración de carpetas ---
input_folder = r'E:\descargas\pose\padel-pose-dataset\imagenes_para_preprocesar'  # Carpeta de imágenes RAW
output_folder = r"E:\descargas\pose\padel-pose-dataset\imagenes_procesadas"      # Carpeta de salida para imágenes procesadas

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# --- Parámetros FLIR ---
emissivity = 0.96
R1, R2, B, F, O = 19839.34, 0.007745727, 1482.6, 1.1, -4096

# --- Procesar todas las imágenes en la carpeta ---
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Solo archivos de imagen
        input_path = os.path.join(input_folder, filename)
        raw_output_path = os.path.join(output_folder, f"raw_{filename}")
        rgb_output_path = os.path.join(output_folder, f"thermal_{filename}")

        # 1. Extraer la imagen térmica en bruto con exiftool
        subprocess.run([
            'exiftool', '-b', '-RawThermalImage',
            input_path
        ], stdout=open(raw_output_path, 'wb'))

        # 2. Cargar la imagen RAW térmica (16 bits)
        raw_img = cv2.imread(raw_output_path, cv2.IMREAD_ANYDEPTH)
        if raw_img is None:
            print(f"Error al cargar {filename}. Saltando...")
            continue

        # 3. Convertir a temperatura (°C)
        raw_corrected = raw_img.astype(np.float32) / emissivity
        temperature_kelvin = B / np.log(R1 / (R2 * (raw_corrected + O)) + F)
        temperature_celsius = temperature_kelvin - 273.15

        # 4. Normalizar a 8 bits (rango 20°C a 40°C para mejor visualización)
        min_temp, max_temp = 20, 40
        normalized = np.clip(temperature_celsius, min_temp, max_temp)
        normalized = ((normalized - min_temp) / (max_temp - min_temp) * 255).astype(np.uint8)

        # 5. Aplicar colormap (Inferno)
        thermal_rgb = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)

        # 6. Guardar la imagen procesada
        cv2.imwrite(rgb_output_path, thermal_rgb)
        print(f"Procesada: {filename} -> {rgb_output_path}")

print("¡Procesamiento completado!")