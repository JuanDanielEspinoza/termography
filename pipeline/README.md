# Pipeline de Segmentación - Dos Modelos YOLO

Pipeline completo que combina **detección de objetos** y **segmentación de instancias** usando dos modelos YOLO en secuencia.

## 📋 Descripción del Flujo

```
Imagen Original
    ↓
[MODELO 1: Detección de Cajas]
    ↓ Crea máscara binaria
[Aplica máscara a imagen] → Imagen con solo región detectada (resto negro)
    ↓
[MODELO 2: Segmentación]
    ↓ Segmenta la región detectada
[Visualiza segmentaciones] → Imagen con máscaras y contornos coloreados
    ↓
Salida Final:
  - imagen_mascara.png (región detectada, resto negro)
  - imagen_segmentacion.png (segmentaciones en colores)
```

## 🚀 Características

✅ **Detección con máscara binaria**: Aísla la región detectada eliminando el ruido de fondo  
✅ **Segmentación en cascada**: Segmenta solo dentro de la región detectada  
✅ **Soporte para imágenes**: Procesa archivos individuales  
✅ **Soporte para directorios**: Procesa múltiples imágenes automáticamente  
✅ **Soporte para videos**: Procesa frame por frame y genera videos de salida  
✅ **Logging detallado**: Monitorea el progreso de procesamiento  
✅ **Modelos personalizables**: Usa tus propios modelos YOLO entrenados  

## 📦 Instalación

### 1. Requisitos Previos
- Python 3.8+
- pip

### 2. Clonar/Descargar el Proyecto

```bash
cd pipeline
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

Las principales librerías instaladas son:
- `ultralytics` - Framework YOLO
- `opencv-python` - Procesamiento de imágenes
- `torch` - Motor de inferencia
- `torchvision` - Visión por computadora

### 4. Crear Carpetas de Entrada/Salida

```bash
mkdir input
mkdir output
```

## 📁 Estructura de Proyecto

```
pipeline/
├── pipeline_segmentation.py      # Script principal (clase PipelineSegmentacion)
├── ejemplos.py                    # Ejemplos de uso
├── requirements.txt               # Dependencias Python
├── README.md                      # Este archivo
│
├── input/                         # Carpeta con imágenes/videos de entrada
│   ├── foto1.jpg
│   ├── foto2.png
│   └── video.mp4
│
├── output/                        # Carpeta con resultados
│   ├── foto1_mascara.png
│   ├── foto1_segmentacion.png
│   ├── video_mascara.mp4
│   └── video_segmentacion.mp4
│
└── models/                        # (Opcional) Modelos personalizados
    ├── detector_personalizado.pt
    └── segmentador_personalizado.pt
```

## 🎯 Uso

### Opción 1: Procesar Una Sola Imagen

```python
from pipeline_segmentation import PipelineSegmentacion

# Crear instancia del pipeline
pipeline = PipelineSegmentacion(
    box_detection_model_path="yolov8m.pt",
    segmentation_model_path="yolov8m-seg.pt",
    output_dir="output",
    confidence_threshold=0.5
)

# Procesar imagen
imagen_mascara, imagen_segmentada = pipeline.procesar_imagen(
    ruta_imagen="input/foto.jpg"
)

# Resultados guardados en:
# - output/foto_mascara.png
# - output/foto_segmentacion.png
```

### Opción 2: Procesar Directorio Completo

```python
pipeline = PipelineSegmentacion(
    box_detection_model_path="yolov8m.pt",
    segmentation_model_path="yolov8m-seg.pt",
    output_dir="output"
)

# Procesa todas las imágenes en input/
pipeline.procesar_directorio("input")
```

### Opción 3: Procesar Video

```python
pipeline = PipelineSegmentacion(
    box_detection_model_path="yolov8m.pt",
    segmentation_model_path="yolov8m-seg.pt",
    output_dir="output"
)

# Procesa video frame por frame
pipeline.procesar_video(
    ruta_video="input/video.mp4",
    nombre_salida="resultado"
)

# Resultados:
# - output/resultado_mascara.mp4
# - output/resultado_segmentacion.mp4
```

### Opción 4: Con Modelos Personalizados

```python
pipeline = PipelineSegmentacion(
    box_detection_model_path="models/mi_detector.pt",
    segmentation_model_path="models/mi_segmentador.pt",
    output_dir="output",
    confidence_threshold=0.6
)

pipeline.procesar_imagen("input/imagen.jpg")
```

## 🔧 Parámetros Configurables

### Constructor de PipelineSegmentacion

```python
pipeline = PipelineSegmentacion(
    box_detection_model_path: str,      # Ruta al modelo de detección YOLO
    segmentation_model_path: str,       # Ruta al modelo de segmentación YOLO
    output_dir: str = "output",         # Directorio de salida
    confidence_threshold: float = 0.5   # Umbral de confianza (0-1)
)
```

### Modelos YOLO Disponibles

#### Para Detección de Cajas:
- `yolov8n.pt` - Nano (rápido, menos preciso) ~3.2M
- `yolov8s.pt` - Small (balance) ~11.2M
- `yolov8m.pt` - Medium (recomendado) ~25.9M
- `yolov8l.pt` - Large (más preciso) ~52.9M
- `yolov8x.pt` - Extra Large (máxima precisión) ~97.9M

#### Para Segmentación:
- `yolov8n-seg.pt` - Nano ~6.7M
- `yolov8s-seg.pt` - Small ~15.5M
- `yolov8m-seg.pt` - Medium ~27.3M
- `yolov8l-seg.pt` - Large ~83.7M
- `yolov8x-seg.pt` - Extra Large ~161M

**Nota**: Los modelos se descargan automáticamente la primera vez que se ejecuta el script.

## 📊 Métodos Disponibles

### procesar_imagen()
```python
imagen_mascara, imagen_segmentada = pipeline.procesar_imagen(
    ruta_imagen: str,           # Ruta a imagen (jpg, png, bmp, etc)
    nombre_salida: str = None   # Nombre para archivos de salida
)
```
Retorna tupla con (imagen_con_mascara, imagen_segmentada)

### procesar_directorio()
```python
pipeline.procesar_directorio(
    ruta_directorio: str,                      # Carpeta con imágenes
    extensiones: list = ['jpg', 'jpeg', 'png'] # Extensiones a procesar
)
```
Procesa todas las imágenes del directorio automáticamente.

### procesar_video()
```python
pipeline.procesar_video(
    ruta_video: str,              # Ruta al archivo de video
    nombre_salida: str = "video"  # Nombre para videos de salida
)
```
Genera dos videos: `*_mascara.mp4` y `*_segmentacion.mp4`

## 🎨 Salidas del Pipeline

### Imagen con Máscara (mascara.png)
- Región detectada: colores originales
- Resto de la imagen: negro (0,0,0)
- Propósito: Enfoque en la región de interés

### Imagen de Segmentación (segmentacion.png)
- Fondo: negro (0,0,0)
- Cada segmento: color único aleatorio con transparencia
- Contornos: línea roja
- Propósito: Visualizar las instancias segmentadas

## 📝 Ejemplo Completo

```python
from pipeline_segmentation import PipelineSegmentacion

# 1. Crear pipeline
pipeline = PipelineSegmentacion(
    box_detection_model_path="yolov8m.pt",
    segmentation_model_path="yolov8m-seg.pt",
    output_dir="resultados",
    confidence_threshold=0.5
)

# 2. Procesar imagen
imagen_m, imagen_s = pipeline.procesar_imagen("fotos/imagen1.jpg")

# 3. Procesar directorio entero
pipeline.procesar_directorio("fotos")

# 4. Procesar video
pipeline.procesar_video("videos/video1.mp4")

# Los resultados estarán en la carpeta "resultados/"
```

## ⚡ Optimizaciones y Tips

### Para Mejor Rendimiento
1. Usa modelos más pequeños (`yolov8n`, `yolov8s`)
2. Reduce la resolución de entrada si es muy alta
3. Procesa en GPU si está disponible (automático con torch+cuda)

### Para Mayor Precisión
1. Usa modelos más grandes (`yolov8l`, `yolov8x`)
2. Aumenta `confidence_threshold` a 0.7-0.9
3. Entrena modelos personalizados con tu dataset

### Procesamiento en Batch
Para procesar muchas imágenes, usa `procesar_directorio()`:
```python
pipeline.procesar_directorio("input")  # Procesa todo automáticamente
```

## 🐛 Solución de Problemas

### "ModuleNotFoundError: No module named 'ultralytics'"
```bash
pip install ultralytics opencv-python
```

### "CUDA out of memory"
Usa modelos más pequeños:
```python
PipelineSegmentacion("yolov8n.pt", "yolov8n-seg.pt")
```

### "No se detectan objetos"
1. Verifica que la imagen sea clara
2. Reduce `confidence_threshold` a 0.3-0.4
3. Asegúrate de usar el modelo correcto para tus objetos

### Los videos salen en blanco/negro
Instala el codec de video:
```bash
pip install opencv-python-headless
```

## 📚 Recursos Adicionales

- **Documentación YOLO**: https://docs.ultralytics.com/
- **Entrenar modelos YOLO**: https://docs.ultralytics.com/tasks/train/
- **YOLOv8 Segmentación**: https://docs.ultralytics.com/tasks/segment/

## 📄 Licencia

Este proyecto es libre para usar y modificar.

## 🤝 Contribuciones

Las mejoras y contribuciones son bienvenidas. Siéntete libre de hacer fork y proponer cambios.

---

**Creado para**: Procesamiento de imágenes con múltiples modelos YOLO en cascada  
**Última actualización**: Diciembre 2024
