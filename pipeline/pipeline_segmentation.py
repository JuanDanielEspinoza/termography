"""
Pipeline de segmentación con dos modelos YOLO:
1. Box Detection: Detección de objetos y creación de máscaras binarias
2. Segmentation: Segmentación de las regiones detectadas

Salida:
- Imagen con máscara binaria aplicada (solo región detectada)
- Imagen con segmentaciones de la región de interés
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from ultralytics import YOLO
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineSegmentacion:
    """
    Pipeline que combina detección de cajas y segmentación de instancias.
    """
    def __init__(
        self,
        box_detection_model_path: str,
        segmentation_model_path: str,
        output_dir: str = "output",
        confidence_threshold: float = 0.5
    ):
        """
        Inicializa el pipeline con los dos modelos YOLO.
        
        Args:
            box_detection_model_path: Ruta al modelo de detección de cajas
            segmentation_model_path: Ruta al modelo de segmentación
            output_dir: Directorio donde guardar las salidas
            confidence_threshold: Umbral de confianza para detecciones
        """
        self.box_detector = YOLO(box_detection_model_path)
        self.segmenter = YOLO(segmentation_model_path)
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold
        
        # Crear subdirectorios de salida
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.mascara_dir = self.output_dir / "mascaras_binarias"
        self.segmentacion_dir = self.output_dir / "segmentaciones"
        self.mascara_dir.mkdir(exist_ok=True, parents=True)
        self.segmentacion_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Pipeline inicializado con modelos:")
        logger.info(f"  - Detector: {box_detection_model_path}")
        logger.info(f"  - Segmentador: {segmentation_model_path}")
        logger.info(f"  - Directorio de salida: {self.output_dir}")
        logger.info(f"  - Máscaras binarias: {self.mascara_dir}")
        logger.info(f"  - Segmentaciones: {self.segmentacion_dir}")
    
    def _crear_mascara_binaria(
        self,
        imagen: np.ndarray,
        detecciones: list
    ) -> np.ndarray:
        """
        Crea una máscara binaria a partir de las detecciones de cajas.
        
        Args:
            imagen: Imagen original
            detecciones: Detecciones del modelo YOLO
            
        Returns:
            Máscara binaria (imagen en escala de grises: 0 y 255)
        """
        altura, ancho = imagen.shape[:2]
        mascara = np.zeros((altura, ancho), dtype=np.uint8)
        
        for deteccion in detecciones:
            # Obtener coordenadas de la caja
            x1, y1, x2, y2 = map(int, deteccion.xyxy[0])
            
            # Asegurar que las coordenadas están dentro de los límites
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(ancho, x2), min(altura, y2)
            
            # Crear región blanca en la máscara
            mascara[y1:y2, x1:x2] = 255
        
        return mascara
    
    def _aplicar_mascara(
        self,
        imagen: np.ndarray,
        mascara: np.ndarray
    ) -> np.ndarray:
        """
        Aplica la máscara binaria a la imagen original.
        
        Args:
            imagen: Imagen original
            mascara: Máscara binaria
            
        Returns:
            Imagen con máscara aplicada (región detectada, resto negro)
        """
        # Normalizar máscara a rango [0, 1]
        mascara_normalizada = mascara.astype(float) / 255.0
        
        # Aplicar máscara a cada canal
        if len(imagen.shape) == 3:
            imagen_mascara = imagen * mascara_normalizada[:, :, np.newaxis]
        else:
            imagen_mascara = imagen * mascara_normalizada
        
        return imagen_mascara.astype(np.uint8)
    def _procesar_segmentaciones(
        self,
        imagen: np.ndarray,
        resultados_segmentacion: list
    ) -> np.ndarray:
        """
        Procesa los resultados de segmentación y crea visualizaciones.
        
        Args:
            imagen: Imagen original (o con máscara aplicada)
            resultados_segmentacion: Resultados del modelo de segmentación
            
        Returns:
            Imagen con máscaras de segmentación dibujadas
        """
        imagen_output = imagen.copy()
        altura, ancho = imagen.shape[:2]
        
        # Colores aleatorios para diferentes instancias segmentadas
        colores = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
        
        for resultado in resultados_segmentacion:
            if resultado.masks is not None:
                # Obtener máscaras de segmentación
                mascaras = resultado.masks.data
                
                logger.info(f"Se encontraron {len(mascaras)} segmentación(es)")
                
                for idx, mascara_inst in enumerate(mascaras):
                    # Convertir tensor a numpy array
                    mascara_np = mascara_inst.cpu().numpy().astype(np.float32)
                    
                    # Redimensionar máscara al tamaño de la imagen
                    mascara_resized = cv2.resize(
                        mascara_np,
                        (ancho, altura),
                        interpolation=cv2.INTER_LINEAR
                    )
                    
                    # Aplicar umbral para obtener máscara binaria
                    _, mascara_binaria = cv2.threshold(
                        (mascara_resized * 255).astype(np.uint8), 
                        128, 
                        255, 
                        cv2.THRESH_BINARY
                    )
                    
                    # Obtener contornos
                    contornos, _ = cv2.findContours(
                        mascara_binaria,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    # Color para esta instancia
                    color = tuple(map(int, colores[idx % len(colores)]))
                      # Dibujar máscaras coloreadas con transparencia mínima
                    mascara_color = np.zeros_like(imagen_output, dtype=np.uint8)
                    cv2.drawContours(mascara_color, contornos, -1, color, -1)
                    
                    # Fusionar con la imagen (aplicar transparencia muy baja)
                    alpha = 0.15
                    imagen_output = cv2.addWeighted(
                        imagen_output,
                        1 - alpha,
                        mascara_color,
                        alpha,
                        0
                    )
                    
                    # Dibujar contornos en blanco para mejor visualización
                    cv2.drawContours(imagen_output, contornos, -1, (255, 255, 255), 2)
        
        return imagen_output
    
    def procesar_imagen(
        self,
        ruta_imagen: str,
        nombre_salida: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Procesa una imagen a través del pipeline completo.
        
        Args:
            ruta_imagen: Ruta a la imagen de entrada
            nombre_salida: Nombre base para los archivos de salida
            
        Returns:
            Tupla con (imagen con máscara, imagen con segmentaciones)
        """
        # Leer imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            logger.error(f"No se pudo cargar la imagen: {ruta_imagen}")
            return None, None
        
        if nombre_salida is None:
            nombre_salida = Path(ruta_imagen).stem
        
        logger.info(f"Procesando imagen: {ruta_imagen}")
        
        # ========== PASO 1: DETECCIÓN DE CAJAS ==========
        logger.info("Paso 1: Detectando cajas...")
        resultados_deteccion = self.box_detector.predict(
            ruta_imagen,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        # Crear máscara binaria
        detecciones = resultados_deteccion[0].boxes
        mascara = self._crear_mascara_binaria(imagen, detecciones)
        
        # Aplicar máscara a la imagen
        imagen_con_mascara = self._aplicar_mascara(imagen, mascara)
        
        logger.info(f"Se detectaron {len(detecciones)} región(es)")
          # ========== PASO 2: SEGMENTACIÓN ==========
        logger.info("Paso 2: Segmentando regiones detectadas...")
        
        # Usar la imagen original para predicciones de segmentación
        resultados_segmentacion = self.segmenter.predict(
            imagen_con_mascara,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        # Procesar segmentaciones y dibujarlas sobre la imagen con máscara
        imagen_segmentada = self._procesar_segmentaciones(
            imagen_con_mascara,
            resultados_segmentacion
        )
        
        # ========== GUARDAR SALIDAS ==========
        logger.info("Guardando resultados...")
        
        # Guardar imagen con máscara aplicada
        ruta_mascara = self.mascara_dir / f"{nombre_salida}_mascara.png"
        cv2.imwrite(str(ruta_mascara), imagen_con_mascara)
        logger.info(f"Imagen con máscara guardada: {ruta_mascara}")
        
        # Guardar imagen con segmentaciones dibujadas
        ruta_segmentacion = self.segmentacion_dir / f"{nombre_salida}_segmentacion.png"
        if imagen_segmentada is not None:
            cv2.imwrite(str(ruta_segmentacion), imagen_segmentada)
            logger.info(f"Imagen de segmentación guardada: {ruta_segmentacion}")
        else:
            logger.warning(f"No se pudo generar segmentación para: {nombre_salida}")
        
        return imagen_con_mascara, imagen_segmentada
    def procesar_directorio(self, ruta_directorio: str, extensiones: list = None):
        """
        Procesa todas las imágenes en un directorio.
        
        Args:
            ruta_directorio: Ruta al directorio con imágenes
            extensiones: Lista de extensiones a procesar (ej: ['jpg', 'png'])
        """
        if extensiones is None:
            extensiones = ['jpg', 'jpeg', 'png', 'bmp']
        
        directorio = Path(ruta_directorio)
        
        if not directorio.exists():
            logger.error(f"Directorio no encontrado: {ruta_directorio}")
            return
        
        # Buscar todas las imágenes
        imagenes = []
        for ext in extensiones:
            imagenes.extend(directorio.glob(f"*.{ext}"))
            imagenes.extend(directorio.glob(f"*.{ext.upper()}"))
        
        # Eliminar duplicados
        imagenes = list(set(imagenes))
        
        if not imagenes:
            logger.warning(f"No se encontraron imágenes en: {ruta_directorio}")
            return
        
        logger.info(f"Se encontraron {len(imagenes)} imagen(es) para procesar")
        logger.info(f"Carpeta de máscaras binarias: {self.mascara_dir}")
        logger.info(f"Carpeta de segmentaciones: {self.segmentacion_dir}")
        
        # Procesar cada imagen
        imagenes_procesadas = 0
        imagenes_error = 0
        
        for idx, ruta_imagen in enumerate(sorted(imagenes), 1):
            try:
                logger.info(f"[{idx}/{len(imagenes)}] Procesando: {ruta_imagen.name}")
                self.procesar_imagen(str(ruta_imagen))
                imagenes_procesadas += 1
            except Exception as e:
                logger.error(f"Error procesando {ruta_imagen.name}: {str(e)}")
                imagenes_error += 1
        
        # Resumen final
        logger.info("=" * 60)
        logger.info("PROCESAMIENTO COMPLETADO")
        logger.info("=" * 60)
        logger.info(f"Total de imágenes: {len(imagenes)}")
        logger.info(f"Imágenes procesadas correctamente: {imagenes_procesadas}")
        logger.info(f"Imágenes con error: {imagenes_error}")
        logger.info(f"Salidas guardadas en:")
        logger.info(f"  - Máscaras: {self.mascara_dir}")
        logger.info(f"  - Segmentaciones: {self.segmentacion_dir}")
    
    def procesar_video(
        self,
        ruta_video: str,
        nombre_salida: str = "video_procesado"
    ):
        """
        Procesa un video frame por frame.
        
        Args:
            ruta_video: Ruta al archivo de video
            nombre_salida: Nombre base para los videos de salida
        """
        cap = cv2.VideoCapture(ruta_video)
        
        if not cap.isOpened():
            logger.error(f"No se pudo abrir el video: {ruta_video}")
            return
        
        # Obtener propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Crear escritores de video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        writer_mascara = cv2.VideoWriter(
            str(self.output_dir / f"{nombre_salida}_mascara.mp4"),
            fourcc,
            fps,
            (ancho, alto)
        )
        
        writer_segmentacion = cv2.VideoWriter(
            str(self.output_dir / f"{nombre_salida}_segmentacion.mp4"),
            fourcc,
            fps,
            (ancho, alto)
        )
        
        frame_count = 0
        logger.info(f"Procesando video: {ruta_video} ({total_frames} frames)")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Procesar frame
            temp_path = "temp_frame.png"
            cv2.imwrite(temp_path, frame)
            
            imagen_mascara, imagen_seg = self.procesar_imagen(
                temp_path,
                f"frame_{frame_count:05d}"
            )
            
            if imagen_mascara is not None and imagen_seg is not None:
                writer_mascara.write(imagen_mascara)
                writer_segmentacion.write(imagen_seg)
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"Procesados {frame_count}/{total_frames} frames")
        
        cap.release()
        writer_mascara.release()
        writer_segmentacion.release()
        
        # Limpiar archivo temporal
        Path(temp_path).unlink(missing_ok=True)
        
        logger.info("Video procesado completamente")


def main():
    """
    Ejemplo de uso del pipeline.
    """
    # ========== CONFIGURACIÓN ==========
    # Reemplaza estas rutas con tus modelos
    MODELO_DETECCION = "box.pt"  # O tu modelo personalizado
    MODELO_SEGMENTACION = "segmentation.pt"  # O tu modelo personalizado
    
    # Rutas de entrada/salida
    DIRECTORIO_ENTRADA = "validate_segmentation"  # Carpeta con las imágenes a procesar
    DIRECTORIO_SALIDA = "output"  # Carpeta donde se guardarán los resultados
    
    # ========== CREAR PIPELINE ==========
    pipeline = PipelineSegmentacion(
        box_detection_model_path=MODELO_DETECCION,
        segmentation_model_path=MODELO_SEGMENTACION,
        output_dir=DIRECTORIO_SALIDA,
        confidence_threshold=0.4
    )
    
    # ========== PROCESAR DIRECTORIO ==========
    # Procesa todas las imágenes en el directorio de entrada
    # Crea automáticamente:
    #   - output/mascaras_binarias/ (imágenes con máscaras aplicadas)
    #   - output/segmentaciones/ (imágenes con segmentaciones encontradas)
    pipeline.procesar_directorio(DIRECTORIO_ENTRADA)


if __name__ == "__main__":
    main()
