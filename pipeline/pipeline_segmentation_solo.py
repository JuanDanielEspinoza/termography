"""
Pipeline de segmentación con modelo YOLO:
- Segmentation: Segmentación directa de instancias en la imagen

Salida:
- Imagen con segmentaciones dibujadas sobre la imagen original
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from ultralytics import YOLO
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineSegmentacionSolo:
    """
    Pipeline que realiza segmentación directa sin detección de cajas.
    """
    def __init__(
        self,
        segmentation_model_path: str,
        output_dir: str = "output",
        confidence_threshold: float = 0.5
    ):
        """
        Inicializa el pipeline con el modelo YOLO de segmentación.
        
        Args:
            segmentation_model_path: Ruta al modelo de segmentación
            output_dir: Directorio donde guardar las salidas
            confidence_threshold: Umbral de confianza para detecciones
        """
        self.segmenter = YOLO(segmentation_model_path)
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold
        
        # Crear directorio de salida
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.segmentacion_dir = self.output_dir / "segmentaciones"
        self.segmentacion_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Pipeline inicializado con modelo:")
        logger.info(f"  - Segmentador: {segmentation_model_path}")
        logger.info(f"  - Directorio de salida: {self.output_dir}")
        logger.info(f"  - Segmentaciones: {self.segmentacion_dir}")
    
    def _procesar_segmentaciones(
        self,
        imagen: np.ndarray,
        resultados_segmentacion: list
    ) -> np.ndarray:
        """
        Procesa los resultados de segmentación y crea visualizaciones.
        
        Args:
            imagen: Imagen original
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
    ) -> np.ndarray:
        """
        Procesa una imagen a través del pipeline de segmentación.
        
        Args:
            ruta_imagen: Ruta a la imagen de entrada
            nombre_salida: Nombre base para los archivos de salida
            
        Returns:
            Imagen con segmentaciones dibujadas
        """
        # Leer imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            logger.error(f"No se pudo cargar la imagen: {ruta_imagen}")
            return None
        
        if nombre_salida is None:
            nombre_salida = Path(ruta_imagen).stem
        
        logger.info(f"Procesando imagen: {ruta_imagen}")
        
        # ========== SEGMENTACIÓN ==========
        logger.info("Realizando segmentación...")
        resultados_segmentacion = self.segmenter.predict(
            ruta_imagen,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        # Procesar segmentaciones y dibujarlas sobre la imagen original
        imagen_segmentada = self._procesar_segmentaciones(
            imagen,
            resultados_segmentacion
        )
        
        # ========== GUARDAR SALIDA ==========
        logger.info("Guardando resultados...")
        
        # Guardar imagen con segmentaciones dibujadas
        ruta_segmentacion = self.segmentacion_dir / f"{nombre_salida}_segmentacion.png"
        if imagen_segmentada is not None:
            cv2.imwrite(str(ruta_segmentacion), imagen_segmentada)
            logger.info(f"Imagen de segmentación guardada: {ruta_segmentacion}")
        else:
            logger.warning(f"No se pudo generar segmentación para: {nombre_salida}")
        
        return imagen_segmentada
    
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
        
        # Crear escritor de video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
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
            
            imagen_seg = self.procesar_imagen(
                temp_path,
                f"frame_{frame_count:05d}"
            )
            
            if imagen_seg is not None:
                writer_segmentacion.write(imagen_seg)
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"Procesados {frame_count}/{total_frames} frames")
        
        cap.release()
        writer_segmentacion.release()
        
        # Limpiar archivo temporal
        Path(temp_path).unlink(missing_ok=True)
        
        logger.info("Video procesado completamente")


def main():
    """
    Ejemplo de uso del pipeline.
    """
    # ========== CONFIGURACIÓN ==========
    # Reemplaza esta ruta con tu modelo
    MODELO_SEGMENTACION = "seg.pt"  # O tu modelo personalizado
    
    # Rutas de entrada/salida
    DIRECTORIO_ENTRADA = "validate_segmentation"  # Carpeta con las imágenes a procesar
    DIRECTORIO_SALIDA = "output"  # Carpeta donde se guardarán los resultados
    
    # ========== CREAR PIPELINE ==========
    pipeline = PipelineSegmentacionSolo(
        segmentation_model_path=MODELO_SEGMENTACION,
        output_dir=DIRECTORIO_SALIDA,
        confidence_threshold=0.7
    )
    
    # ========== PROCESAR DIRECTORIO ==========
    # Procesa todas las imágenes en el directorio de entrada
    # Crea automáticamente:
    #   - output/segmentaciones/ (imágenes con segmentaciones encontradas)
    pipeline.procesar_directorio(DIRECTORIO_ENTRADA)


if __name__ == "__main__":
    main()
