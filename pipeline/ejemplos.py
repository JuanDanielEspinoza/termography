"""
Ejemplos de uso del pipeline de segmentación con dos modelos YOLO
"""

from pipeline_segmentation import PipelineSegmentacion
import logging

# Configurar logging para ver todos los detalles
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ============================================================================
# EJEMPLO 1: Procesar una sola imagen
# ============================================================================
def ejemplo_procesar_imagen():
    """Procesa una única imagen con el pipeline."""
    
    pipeline = PipelineSegmentacion(
        box_detection_model_path="yolov8m.pt",  # Tu modelo de detección
        segmentation_model_path="yolov8m-seg.pt",  # Tu modelo de segmentación
        output_dir="output",
        confidence_threshold=0.5
    )
    
    # Procesar imagen
    imagen_mascara, imagen_segmentada = pipeline.procesar_imagen(
        ruta_imagen="input/foto.jpg",
        nombre_salida="resultado_foto"
    )
    
    print("✓ Imagen procesada correctamente")
    print("  - Guardado: output/resultado_foto_mascara.png")
    print("  - Guardado: output/resultado_foto_segmentacion.png")


# ============================================================================
# EJEMPLO 2: Procesar múltiples imágenes de un directorio
# ============================================================================
def ejemplo_procesar_directorio():
    """Procesa todas las imágenes en un directorio."""
    
    pipeline = PipelineSegmentacion(
        box_detection_model_path="yolov8m.pt",
        segmentation_model_path="yolov8m-seg.pt",
        output_dir="output",
        confidence_threshold=0.5
    )
    
    # Procesar directorio completo
    pipeline.procesar_directorio(
        ruta_directorio="input",
        extensiones=['jpg', 'jpeg', 'png']
    )
    
    print("✓ Directorio procesado")
    print("  Los resultados están en 'output/'")


# ============================================================================
# EJEMPLO 3: Procesar un video
# ============================================================================
def ejemplo_procesar_video():
    """Procesa un video frame por frame."""
    
    pipeline = PipelineSegmentacion(
        box_detection_model_path="yolov8m.pt",
        segmentation_model_path="yolov8m-seg.pt",
        output_dir="output",
        confidence_threshold=0.5
    )
    
    # Procesar video
    pipeline.procesar_video(
        ruta_video="input/video.mp4",
        nombre_salida="resultado_video"
    )
    
    print("✓ Video procesado correctamente")
    print("  - Guardado: output/resultado_video_mascara.mp4")
    print("  - Guardado: output/resultado_video_segmentacion.mp4")


# ============================================================================
# EJEMPLO 4: Usar modelos personalizados entrenados
# ============================================================================
def ejemplo_modelos_personalizados():
    """Usa modelos YOLO personalizados entrenados."""
    
    pipeline = PipelineSegmentacion(
        box_detection_model_path="models/mi_modelo_deteccion.pt",
        segmentation_model_path="models/mi_modelo_segmentacion.pt",
        output_dir="output_personalizado",
        confidence_threshold=0.6  # Umbral más alto para mayor precisión
    )
    
    # Procesar imagen
    imagen_mascara, imagen_segmentada = pipeline.procesar_imagen(
        ruta_imagen="input/imagen_especial.jpg"
    )
    
    print("✓ Procesado con modelos personalizados")


# ============================================================================
# ESTRUCTURA DE CARPETAS RECOMENDADA
# ============================================================================
"""
pipeline/
├── pipeline_segmentation.py      # Script principal del pipeline
├── ejemplos.py                    # Este archivo
├── requirements.txt               # Dependencias
├── input/                         # Carpeta con imágenes/videos de entrada
│   ├── foto1.jpg
│   ├── foto2.png
│   └── video.mp4
├── output/                        # Carpeta con resultados
│   ├── foto1_mascara.png
│   ├── foto1_segmentacion.png
│   ├── foto2_mascara.png
│   ├── foto2_segmentacion.png
│   ├── video_mascara.mp4
│   └── video_segmentacion.mp4
└── models/                        # (Opcional) Modelos personalizados
    ├── mi_modelo_deteccion.pt
    └── mi_modelo_segmentacion.pt
"""


# ============================================================================
# INSTRUCCIONES DE INSTALACIÓN
# ============================================================================
"""
1. Crear carpeta del proyecto:
   mkdir pipeline
   cd pipeline

2. Crear carpetas de entrada y salida:
   mkdir input
   mkdir output
   mkdir models

3. Instalar dependencias:
   pip install -r requirements.txt

4. Colocar imágenes/videos en la carpeta 'input/'

5. Actualizar las rutas en los ejemplos según tus archivos

6. Ejecutar uno de los ejemplos:
   python ejemplos.py
"""


# ============================================================================
# PARÁMETROS PERSONALIZABLES
# ============================================================================
"""
PARÁMETROS DEL PIPELINE:

1. confidence_threshold (float entre 0 y 1):
   - Umbral de confianza para las detecciones
   - Valores altos = menos detecciones pero más confiables
   - Recomendado: 0.5 (por defecto)

2. box_detection_model_path:
   - "yolov8n.pt" = Modelo nano (rápido, menos preciso)
   - "yolov8m.pt" = Modelo mediano (balance)
   - "yolov8l.pt" = Modelo grande (más preciso, más lento)
   - "yolov8x.pt" = Modelo extra grande (máxima precisión)
   - O la ruta a tu modelo personalizado

3. segmentation_model_path:
   - Debe ser un modelo YOLO con capacidad de segmentación
   - "yolov8m-seg.pt", "yolov8l-seg.pt", etc.
"""


if __name__ == "__main__":
    # Descomenta el ejemplo que quieras ejecutar:
    
    # ejemplo_procesar_imagen()
    # ejemplo_procesar_directorio()
    # ejemplo_procesar_video()
    # ejemplo_modelos_personalizados()
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  Pipeline de Segmentación de Dos Modelos YOLO             ║
    ║                                                            ║
    ║  Para usar este pipeline, descomenta uno de los ejemplos  ║
    ║  en la función main() de este archivo.                    ║
    ║                                                            ║
    ║  Ver documentación en pipeline_segmentation.py            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
