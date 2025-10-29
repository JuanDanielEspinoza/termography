import os
import cv2
import numpy as np
from pathlib import Path
import shutil

# Configuración
carpeta_entrada = r"C:\Users\ASUS\Desktop\Canada_Repository\termography\data\data_roboflow_flir_one_160_120"
carpeta_salida = r"C:\Users\ASUS\Desktop\Canada_Repository\termography\data\processed_thermal_visual"

def install_opencv():
    """Instalar OpenCV si no está disponible"""
    try:
        import cv2
        return True
    except ImportError:
        print("📦 Instalando OpenCV...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
            import cv2
            return True
        except:
            print("❌ No se pudo instalar OpenCV")
            return False

def process_thermal_visual(image_path, output_dir):
    """Procesar imagen térmica basándose en características visuales"""
    
    if not install_opencv():
        print("   ❌ OpenCV no disponible, copiando imagen original")
        shutil.copy2(image_path, output_dir)
        return False
    
    import cv2
    import numpy as np
    
    nombre_base = os.path.splitext(os.path.basename(image_path))[0]
    
    # Leer imagen
    img = cv2.imread(image_path)
    if img is None:
        print(f"   ❌ No se pudo leer la imagen")
        return False
    
    print(f"   📐 Dimensiones: {img.shape}")
    
    # 1. Guardar imagen original redimensionada
    img_resized = cv2.resize(img, (160, 120))  # FLIR One resolución
    cv2.imwrite(os.path.join(output_dir, f"{nombre_base}_160x120.jpg"), img_resized)
    
    # 2. Convertir a escala de grises (simula datos térmicos)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, f"{nombre_base}_grayscale.jpg"), gray)
    
    # 3. Aplicar mapa de calor (colormap térmico)
    thermal_colormap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir, f"{nombre_base}_thermal_jet.jpg"), thermal_colormap)
    
    # 4. Aplicar otro mapa térmico
    thermal_inferno = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(output_dir, f"{nombre_base}_thermal_inferno.jpg"), thermal_inferno)
    
    # 5. Ecualización de histograma (mejora contraste)
    equalized = cv2.equalizeHist(gray)
    cv2.imwrite(os.path.join(output_dir, f"{nombre_base}_equalized.jpg"), equalized)
    
    # 6. Análisis de regiones calientes/frías
    # Umbralización para destacar regiones
    _, thresh_hot = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    _, thresh_cold = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
    
    cv2.imwrite(os.path.join(output_dir, f"{nombre_base}_hot_regions.jpg"), thresh_hot)
    cv2.imwrite(os.path.join(output_dir, f"{nombre_base}_cold_regions.jpg"), thresh_cold)
    
    # 7. Crear información de análisis
    analysis_info = {
        'mean_intensity': np.mean(gray),
        'std_intensity': np.std(gray),
        'min_intensity': np.min(gray),
        'max_intensity': np.max(gray),
        'dynamic_range': np.max(gray) - np.min(gray),
        'hot_pixels_count': np.sum(thresh_hot == 255),
        'cold_pixels_count': np.sum(thresh_cold == 255),
        'total_pixels': gray.shape[0] * gray.shape[1]
    }
    
    # Guardar información de análisis
    info_file = os.path.join(output_dir, f"{nombre_base}_thermal_analysis.txt")
    with open(info_file, 'w') as f:
        f.write(f"ANÁLISIS TÉRMICO VISUAL - {nombre_base}\n")
        f.write(f"={'='*50}\n\n")
        f.write(f"Intensidad promedio: {analysis_info['mean_intensity']:.2f}\n")
        f.write(f"Desviación estándar: {analysis_info['std_intensity']:.2f}\n")
        f.write(f"Rango dinámico: {analysis_info['dynamic_range']}\n")
        f.write(f"Píxeles calientes (>180): {analysis_info['hot_pixels_count']}\n")
        f.write(f"Píxeles fríos (<75): {analysis_info['cold_pixels_count']}\n")
        f.write(f"Total de píxeles: {analysis_info['total_pixels']}\n")
        f.write(f"Porcentaje caliente: {(analysis_info['hot_pixels_count']/analysis_info['total_pixels'])*100:.2f}%\n")
        f.write(f"Porcentaje frío: {(analysis_info['cold_pixels_count']/analysis_info['total_pixels'])*100:.2f}%\n")
    
    print(f"   ✅ Procesamiento visual completado")
    print(f"   📊 Rango dinámico: {analysis_info['dynamic_range']}")
    print(f"   🌡️ Intensidad promedio: {analysis_info['mean_intensity']:.1f}")
    
    return True

# Crear carpeta de salida
Path(carpeta_salida).mkdir(parents=True, exist_ok=True)

print("🎨 PROCESAMIENTO VISUAL DE IMÁGENES TÉRMICAS")
print(f"📁 Entrada: {carpeta_entrada}")
print(f"📁 Salida: {carpeta_salida}")
print("=" * 60)

# Obtener imágenes
archivos_jpg = [f for f in os.listdir(carpeta_entrada) if f.lower().endswith(('.jpg', '.jpeg'))]

if not archivos_jpg:
    print("❌ No se encontraron imágenes")
    exit(1)

print(f"📊 Encontradas {len(archivos_jpg)} imágenes")
print(f"🎯 Procesando las primeras 5 imágenes como muestra...\n")

procesadas = 0
for i, archivo in enumerate(archivos_jpg[:5]):
    print(f"📷 [{i+1}/5] Procesando: {archivo}")
    
    ruta_entrada = os.path.join(carpeta_entrada, archivo)
    
    # Crear subcarpeta para cada imagen
    img_output_dir = os.path.join(carpeta_salida, os.path.splitext(archivo)[0])
    Path(img_output_dir).mkdir(parents=True, exist_ok=True)
    
    if process_thermal_visual(ruta_entrada, img_output_dir):
        procesadas += 1

print(f"\n🎉 PROCESAMIENTO COMPLETADO")
print(f"   ✅ {procesadas}/5 imágenes procesadas")
print(f"   📁 Resultados en: {carpeta_salida}")

print(f"\n📋 ARCHIVOS GENERADOS POR IMAGEN:")
print(f"   • *_160x120.jpg - Imagen redimensionada a resolución FLIR")
print(f"   • *_grayscale.jpg - Escala de grises (simula datos térmicos)")
print(f"   • *_thermal_jet.jpg - Mapa de calor JET")
print(f"   • *_thermal_inferno.jpg - Mapa de calor INFERNO")
print(f"   • *_equalized.jpg - Contraste mejorado") 
print(f"   • *_hot_regions.jpg - Regiones calientes")
print(f"   • *_cold_regions.jpg - Regiones frías")
print(f"   • *_thermal_analysis.txt - Análisis estadístico")

print(f"\n💡 SIGUIENTE PASO:")
print(f"   Aunque no tengas los datos térmicos originales, puedes:")
print(f"   1. 🧠 Entrenar modelos con las imágenes visuales procesadas")
print(f"   2. 🎨 Usar los mapas de calor generados como pseudo-datos térmicos") 
print(f"   3. 📊 Analizar las estadísticas térmicas visuales")
print(f"   4. 🔄 Aplicar este procesamiento a todo el dataset si funciona bien")