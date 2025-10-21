import os
import shutil
import random
from pathlib import Path

# ==================== CONFIGURACIÓN ====================
# Carpetas de entrada
carpeta_imagenes = r"C:\Users\ASUS\Desktop\Canada\Yolo_FCV_neonatos\images"
carpeta_etiquetas = r"C:\Users\ASUS\Desktop\Canada\Yolo_FCV_neonatos\labels"

# Carpeta de salida
carpeta_salida = r"C:\Users\ASUS\Desktop\Canada\dataset_split"

# Porcentajes de división (deben sumar 100)
PORCENTAJE_TRAIN = 70  # %
PORCENTAJE_VAL = 25    # %
PORCENTAJE_TEST = 5   # %

# Semilla aleatoria (para reproducibilidad)
SEED = 42

# =======================================================

# Validar que los porcentajes suman 100
total = PORCENTAJE_TRAIN + PORCENTAJE_VAL + PORCENTAJE_TEST
if total != 100:
    print(f"❌ Error: Los porcentajes deben sumar 100. Actualmente suman: {total}%")
    exit()

print("📊 División del dataset YOLO")
print(f"   Train: {PORCENTAJE_TRAIN}%")
print(f"   Val: {PORCENTAJE_VAL}%")
print(f"   Test: {PORCENTAJE_TEST}%")
print()

# Verificar que existan las carpetas de entrada
if not os.path.exists(carpeta_imagenes):
    print(f"❌ Error: No existe la carpeta de imágenes: {carpeta_imagenes}")
    exit()

if not os.path.exists(carpeta_etiquetas):
    print(f"❌ Error: No existe la carpeta de etiquetas: {carpeta_etiquetas}")
    exit()

# Crear estructura de carpetas de salida
splits = ['train', 'val', 'test']
for split in splits:
    Path(os.path.join(carpeta_salida, split, 'images')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(carpeta_salida, split, 'labels')).mkdir(parents=True, exist_ok=True)

# Obtener lista de imágenes (sin extensión)
extensiones_imagen = ['.jpg', '.jpeg', '.png', '.bmp']
archivos_imagen = []

for archivo in os.listdir(carpeta_imagenes):
    nombre, ext = os.path.splitext(archivo)
    if ext.lower() in extensiones_imagen:
        # Verificar que existe la etiqueta correspondiente
        archivo_etiqueta = nombre + '.txt'
        ruta_etiqueta = os.path.join(carpeta_etiquetas, archivo_etiqueta)
        
        if os.path.exists(ruta_etiqueta):
            archivos_imagen.append((archivo, archivo_etiqueta))
        else:
            print(f"⚠️ Advertencia: No se encontró etiqueta para {archivo}")

total_archivos = len(archivos_imagen)
print(f"📁 Total de imágenes con etiquetas: {total_archivos}\n")

if total_archivos == 0:
    print("❌ Error: No se encontraron imágenes con etiquetas correspondientes")
    exit()

# Mezclar aleatoriamente
random.seed(SEED)
random.shuffle(archivos_imagen)

# Calcular cantidad de archivos por conjunto
num_train = int(total_archivos * PORCENTAJE_TRAIN / 100)
num_val = int(total_archivos * PORCENTAJE_VAL / 100)
num_test = total_archivos - num_train - num_val  # El resto para test

# Dividir la lista
train_files = archivos_imagen[:num_train]
val_files = archivos_imagen[num_train:num_train + num_val]
test_files = archivos_imagen[num_train + num_val:]

print(f"📊 Distribución:")
print(f"   Train: {len(train_files)} archivos ({len(train_files)/total_archivos*100:.1f}%)")
print(f"   Val: {len(val_files)} archivos ({len(val_files)/total_archivos*100:.1f}%)")
print(f"   Test: {len(test_files)} archivos ({len(test_files)/total_archivos*100:.1f}%)")
print()

# Función para copiar archivos
def copiar_archivos(lista_archivos, split_name):
    print(f"📋 Copiando archivos a {split_name}...")
    for img, lbl in lista_archivos:
        # Copiar imagen
        src_img = os.path.join(carpeta_imagenes, img)
        dst_img = os.path.join(carpeta_salida, split_name, 'images', img)
        shutil.copy2(src_img, dst_img)
        
        # Copiar etiqueta
        src_lbl = os.path.join(carpeta_etiquetas, lbl)
        dst_lbl = os.path.join(carpeta_salida, split_name, 'labels', lbl)
        shutil.copy2(src_lbl, dst_lbl)
    print(f"   ✅ {len(lista_archivos)} archivos copiados\n")

# Copiar archivos a sus carpetas correspondientes
copiar_archivos(train_files, 'train')
copiar_archivos(val_files, 'val')
copiar_archivos(test_files, 'test')

print(f"🎉 ¡Proceso completado!")
print(f"📁 Dataset dividido guardado en: {carpeta_salida}")
print()
print("📂 Estructura de carpetas creada:")
print(f"   {carpeta_salida}/")
print(f"   ├── train/")
print(f"   │   ├── images/")
print(f"   │   └── labels/")
print(f"   ├── val/")
print(f"   │   ├── images/")
print(f"   │   └── labels/")
print(f"   └── test/")
print(f"       ├── images/")
print(f"       └── labels/")
