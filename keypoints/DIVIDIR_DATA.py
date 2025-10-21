import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1, random_seed=42):
    """
    Divide un dataset en train/valid/test, manteniendo la correspondencia entre imágenes y etiquetas.
    
    Args:
        images_dir (str): Directorio con las imágenes.
        labels_dir (str): Directorio con las etiquetas.
        output_dir (str): Directorio de salida donde se crearán train/valid/test.
        train_ratio (float): Proporción para entrenamiento (ej: 0.7).
        valid_ratio (float): Proporción para validación (ej: 0.2).
        test_ratio (float): Proporción para prueba (ej: 0.1).
        random_seed (int): Semilla para reproducibilidad.
    """
    # Verificar que las proporciones sumen 1
    assert abs((train_ratio + valid_ratio + test_ratio) - 1.0) < 1e-9, "Las proporciones deben sumar 1.0"
    
    # Obtener lista de imágenes (sin extensión)
    image_files = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Dividir en train, valid y test
    train_files, test_valid_files = train_test_split(image_files, train_size=train_ratio, random_state=random_seed)
    valid_files, test_files = train_test_split(test_valid_files, test_size=test_ratio/(valid_ratio + test_ratio), random_state=random_seed)
    
    # Crear directorios de salida
    splits = {
        'train': train_files,
        'val': valid_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        # Crear carpetas images y labels para cada split
        os.makedirs(os.path.join(output_dir, split_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split_name, 'labels'), exist_ok=True)
        
        # Copiar imágenes y etiquetas
        for file in files:
            # Copiar imagen (buscar extensión correcta)
            for ext in ['.jpg', '.png', '.jpeg']:
                src_image = os.path.join(images_dir, file + ext)
                if os.path.exists(src_image):
                    shutil.copy(src_image, os.path.join(output_dir, split_name, 'images', file + ext))
                    break
            
            # Copiar etiqueta (.txt)
            src_label = os.path.join(labels_dir, file + '.txt')
            if os.path.exists(src_label):
                shutil.copy(src_label, os.path.join(output_dir, split_name, 'labels', file + '.txt'))
            else:
                print(f"⚠️ Advertencia: No se encontró la etiqueta para {file}")

    print("✅ Dataset dividido correctamente en train/valid/test.")

# Ejemplo de uso
if __name__ == "__main__":
    # Configuración (cambia estas rutas según tu caso)
    IMAGES_DIR = "E:\descargas\pose\padel-pose-dataset\YOLO\imagenes"   # Directorio de imágenes de entrada
    LABELS_DIR = "E:\descargas\pose\padel-pose-dataset\YOLO\labels"   # Directorio de etiquetas de entrada
    OUTPUT_DIR = "E:\descargas\pose\padel-pose-dataset\YOLO"         # Directorio de salida
    
    # Crear las particiones (70% train, 20% valid, 10% test)
    split_dataset(IMAGES_DIR, LABELS_DIR, OUTPUT_DIR, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1)