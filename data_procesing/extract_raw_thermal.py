import os
import subprocess
from pathlib import Path

# Configuración
carpeta_entrada = r"C:\Users\ASUS\Desktop\Canada_Repository\termography\data\data_roboflow_flir_one_160_120"
carpeta_salida = r"C:\Users\ASUS\Desktop\Canada_Repository\termography\data\data_roboflow_flir_one_160_120\extracted_thermal_images"

# Crear carpeta de salida si no existe
Path(carpeta_salida).mkdir(parents=True, exist_ok=True)

archivos_procesados = 0
archivos_error = 0

for archivo in os.listdir(carpeta_entrada):
    if not archivo.lower().endswith(('.jpg', '.jpeg')):
        continue

    ruta_entrada = os.path.join(carpeta_entrada, archivo)
    nombre_base = os.path.splitext(archivo)[0]
    ruta_salida_raw = os.path.join(carpeta_salida, f"{nombre_base}_raw.png")
    ruta_salida_embedded = os.path.join(carpeta_salida, f"{nombre_base}_embedded.png")

    print(f"📷 Procesando: {archivo}")

    try:
        # 1️⃣ Intentar extraer RawThermalImage (para imágenes térmicas puras)
        result = subprocess.run(
            ["exiftool", "-b", "-RawThermalImage", ruta_entrada],
            stdout=open(ruta_salida_raw, "wb"),
            stderr=subprocess.PIPE
        )

        if result.returncode == 0 and os.path.getsize(ruta_salida_raw) > 0:
            print(f"   ✅ Imagen RAW radiométrica guardada como: {os.path.basename(ruta_salida_raw)}")
            archivos_procesados += 1
            continue  # Pasar a la siguiente imagen

        # 2️⃣ Si falla o está vacía, intentar extraer EmbeddedImage (MSX)
        result_emb = subprocess.run(
            ["exiftool", "-b", "-EmbeddedImage", ruta_entrada],
            stdout=open(ruta_salida_embedded, "wb"),
            stderr=subprocess.PIPE
        )

        if result_emb.returncode == 0 and os.path.getsize(ruta_salida_embedded) > 0:
            print(f"   ⚠️ Imagen MSX detectada — EmbeddedImage guardada como: {os.path.basename(ruta_salida_embedded)}")
            archivos_procesados += 1
        else:
            print(f"   ❌ No se pudo extraer ni RAW ni EmbeddedImage de {archivo}")
            archivos_error += 1

    except Exception as e:
        print(f"   ❌ Error inesperado con {archivo}: {str(e)}")
        archivos_error += 1

print(f"\n🎉 Proceso completado:")
print(f"   ✅ {archivos_procesados} imágenes procesadas correctamente")
if archivos_error > 0:
    print(f"   ❌ {archivos_error} imágenes con errores")
print(f"   📁 Imágenes guardadas en: {carpeta_salida}")
