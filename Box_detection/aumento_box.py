
#input_images = "sets/train/images"   # Carpeta con imágenes originales (.jpg/.png)
#input_labels = "sets/train/labels"   # Carpeta con etiquetas YOLO (.txt)
#output_images = "sets_aumented/train/images"
#output_labels = "sets_aumented/train/labels"

import os
import glob
import shutil
from tqdm import tqdm
import cv2
import albumentations as A

# -------- CONFIG ----------
INPUT_DIR = "sets/train/images"      # carpeta con imágenes originales
LABELS_DIR = "sets/train/labels"     # carpeta con .txt YOLO (same basename)
OUTPUT_DIR = "sets_aumented"           # carpeta raíz de salida
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "train/images")
OUTPUT_LABELS_DIR = os.path.join(OUTPUT_DIR, "train/labels")

NUM_AUGS = 5                          # cantidad de aumentos distintos por imagen
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")
MIN_VISIBILITY = 0.3                  # albumentations: elimina bboxes con visibilidad < esto
REPRODUCIBLE = False                  # si True fija semilla para reproducibilidad
SEED = 42
# ---------------------------

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

def find_images(input_dir):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    files.sort()
    return files

def read_yolo_label(label_path):
    """
    Lee un archivo YOLO y devuelve dos listas:
    - bboxes: [(x_center, y_center, w, h), ...] (floats, normalizados 0..1)
    - labels: [class_id_str, ...]
    """
    bboxes = []
    labels = []
    if not os.path.exists(label_path):
        return bboxes, labels
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = parts[0]
            coords = tuple(map(float, parts[1:5]))
            bboxes.append(coords)
            labels.append(cls)
    return bboxes, labels

def write_yolo_label(label_path, bboxes, labels):
    """
    Escribe bboxes y labels en formato YOLO (clase x y w h) con 6 decimales
    """
    with open(label_path, "w") as f:
        for cls, bbox in zip(labels, bboxes):
            x,y,w,h = bbox
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# Si quieres reproducibilidad
if REPRODUCIBLE:
    A.seed(SEED)

# Definimos 5 transformaciones distintas (ajusta parámetros si quieres)
transforms = [
    A.Compose([
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(p=0.8),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=MIN_VISIBILITY)),
    
    A.Compose([
        A.Affine(translate_percent=(0.06, 0.06), scale=(0.9, 1.1), rotate=(-15, 15), p=1.0),
        A.GaussNoise(p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=MIN_VISIBILITY)),
    
    A.Compose([
        # OJO: RandomSizedBBoxSafeCrop puede cambiar tamaño; ajusta (h,w) al deseado
        A.RandomSizedBBoxSafeCrop(height=640, width=640, p=1.0),
        A.HueSaturationValue(p=0.7),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=MIN_VISIBILITY)),
    
    A.Compose([
        A.Rotate(limit=25, p=1.0),
        A.Blur(blur_limit=3, p=0.4),
        A.RandomBrightnessContrast(p=0.7),   # reemplazo de RandomContrast
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=MIN_VISIBILITY)),
    
    A.Compose([
        A.CLAHE(p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=0.8),
        A.Affine(shear=10, scale=(0.9, 1.1), translate_percent=(0.03,0.03), p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=MIN_VISIBILITY)),
]

images = find_images(INPUT_DIR)
if not images:
    raise SystemExit(f"No se encontraron imágenes en {INPUT_DIR} con extensiones {IMAGE_EXTS}")

for img_path in tqdm(images, desc="Procesando imágenes"):
    basename = os.path.splitext(os.path.basename(img_path))[0]
    ext = os.path.splitext(os.path.basename(img_path))[1]  # conserva extensión original
    # Leer imagen (BGR)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: no se pudo leer {img_path}, se salta.")
        continue

    # Leer labels YOLO
    label_path = os.path.join(LABELS_DIR, f"{basename}.txt")
    bboxes, labels = read_yolo_label(label_path)

    # 1) Guardar la imagen original en la carpeta de salida (misma extensión)
    out_img_path = os.path.join(OUTPUT_IMAGES_DIR, f"{basename}{ext}")
    cv2.imwrite(out_img_path, img)

    # Copiar o crear el .txt original en la carpeta de labels
    out_label_path = os.path.join(OUTPUT_LABELS_DIR, f"{basename}.txt")
    if os.path.exists(label_path):
        shutil.copyfile(label_path, out_label_path)
    else:
        open(out_label_path, "w").close()

    # 2) Aplicar cada una de las N transformaciones (una por variante)
    for i in range(NUM_AUGS):
        transform = transforms[i % len(transforms)]
        try:
            augmented = transform(image=img, bboxes=bboxes, class_labels=labels)
        except Exception as e:
            print(f"Error al aplicar transform a {basename}: {e}")
            continue

        aug_img = augmented["image"]
        aug_bboxes = augmented.get("bboxes", [])
        aug_labels = augmented.get("class_labels", [])

        # Guardar imagen aumentada con sufijo (conserva extensión)
        out_aug_img_path = os.path.join(OUTPUT_IMAGES_DIR, f"{basename}_aug{i+1}{ext}")
        cv2.imwrite(out_aug_img_path, aug_img)

        # Guardar labels aumentadas (en formato YOLO)
        out_aug_label_path = os.path.join(OUTPUT_LABELS_DIR, f"{basename}_aug{i+1}.txt")
        if aug_bboxes:
            write_yolo_label(out_aug_label_path, aug_bboxes, aug_labels)
        else:
            # si no quedaron bboxes, creamos archivo vacío
            open(out_aug_label_path, "w").close()

print("Listo. Aumentaciones guardadas en:")
print(" - imágenes:", OUTPUT_IMAGES_DIR)
print(" - etiquetas:", OUTPUT_LABELS_DIR)
