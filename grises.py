import os
from PIL import Image

def convertir_a_grises(folder_entrada, folder_salida):
    """
    Convierte todas las imágenes de un folder a escala de grises
    
    Args:
        folder_entrada (str): Ruta del folder con las imágenes originales
        folder_salida (str): Ruta del folder donde guardar las imágenes en grises
    """
    
    # Crear el folder de salida si no existe
    if not os.path.exists(folder_salida):
        os.makedirs(folder_salida)
    
    # Extensiones de imagen soportadas
    extensiones_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    # Obtener lista de archivos de imagen
    archivos_imagen = [f for f in os.listdir(folder_entrada) 
                      if f.lower().endswith(extensiones_validas)]
    
    print(f"Encontradas {len(archivos_imagen)} imágenes para convertir...")
    
    for i, archivo in enumerate(archivos_imagen, 1):
        try:
            # Ruta completa del archivo de entrada
            ruta_entrada = os.path.join(folder_entrada, archivo)
            
            # Abrir la imagen
            imagen = Image.open(ruta_entrada)
            
            # Convertir a escala de grises
            imagen_gris = imagen.convert('L')
            
            # Crear nombre del archivo de salida
            nombre, extension = os.path.splitext(archivo)
            nombre_salida = f"{nombre}_gris{extension}"
            ruta_salida = os.path.join(folder_salida, nombre_salida)
            
            # Guardar la imagen en escala de grises
            imagen_gris.save(ruta_salida)
            
            print(f"({i}/{len(archivos_imagen)}) Convertida: {archivo} -> {nombre_salida}")
            
        except Exception as e:
            print(f"Error al procesar {archivo}: {str(e)}")
    
    print("¡Conversión completada!")

# Ejemplo de uso
if __name__ == "__main__":
    # Cambiar estas rutas por las tuyas
    folder_imagenes = r"C:\Users\ASUS\Desktop\Canada_Repository\termography\data\sujeto01_imagenes_termicas"
    folder_grises = r"C:\Users\ASUS\Desktop\Canada_Repository\termography\data\imagenes_grises_sujeto_01"
    
    convertir_a_grises(folder_imagenes, folder_grises)