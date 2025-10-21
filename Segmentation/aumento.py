import cv2
import albumentations as A
import os
from glob import glob
import numpy as np

# === Función para leer etiquetas YOLO-Seg ===
def leer_etiqueta_seg_yolo(ruta_txt):
    class_ids, polygons = [], []
    with open(ruta_txt, 'r') as f:
        for linea in f:
            vals = list(map(float, linea.strip().split()))
            if len(vals) < 3:
                continue
            cls = int(vals[0])
            coords = vals[1:]
            poly = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            class_ids.append(cls)
            polygons.append(poly)
    return class_ids, polygons

# === Guardar etiquetas YOLO-Seg ===
def guardar_etiqueta_seg_yolo(ruta_salida, class_ids, polygons):
    with open(ruta_salida, "w") as f:
        for cls, poly in zip(class_ids, polygons):
            linea = [str(cls)]
            for x, y in poly:
                linea += [f"{x:.6f}", f"{y:.6f}"]
            f.write(" ".join(linea) + "\n")

# === Convierte polígonos normalizados a máscara multiclasé ===
def polys_a_mask(polygons, class_ids, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    for cls, poly in zip(class_ids, polygons):
        pts = np.array([[int(x * w), int(y * h)] for x, y in poly], dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=cls+1)
    return mask

# === Extrae polígonos normalizados de una máscara multiclasé ===
def mask_a_polys(mask):
    class_ids, polygons = [], []
    for v in np.unique(mask):
        if v == 0:
            continue
        bin_mask = (mask == v).astype(np.uint8)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 3:
                continue
            h, w = mask.shape
            poly = [(pt[0][0]/w, pt[0][1]/h) for pt in cnt]
            class_ids.append(int(v-1))
            polygons.append(poly)
    return class_ids, polygons

# === Rutas de carpetas ===
carpeta_imagenes        = r"sets/train/images"
carpeta_etiquetas       = r"sets/train/labels"
carpeta_salida_imagenes = r"sets_aumented/train/images"
carpeta_salida_etiquetas= r"sets_aumented/train/labels"
os.makedirs(carpeta_salida_imagenes, exist_ok=True)
os.makedirs(carpeta_salida_etiquetas, exist_ok=True)

# === Lista de aumentos independientes ===
augmentations = [
    ("scale_rotate", A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent={"x":(-0.1,0.1),"y":(-0.1,0.1)}, p=1.0)),
    ("brightness_contrast", A.RandomBrightnessContrast(p=1.0)),
    ("rotate90", A.RandomRotate90(p=1.0)),
    ("hflip", A.HorizontalFlip(p=1.0)),
    ("vflip", A.VerticalFlip(p=1.0)),
    ("grid_distortion", A.GridDistortion(p=1.0)),
    ("brightness_contrast2", A.RandomBrightnessContrast(p=1.0))

]

# === Procesamiento ===
for ruta_img in glob(os.path.join(carpeta_imagenes, "*.jpg")):
    base     = os.path.splitext(os.path.basename(ruta_img))[0]
    ruta_lbl = os.path.join(carpeta_etiquetas, base + ".txt")
    if not os.path.exists(ruta_lbl):
        print(f"⚠️ Sin etiqueta para {base}, se omite.")
        continue

    img = cv2.imread(ruta_img)
    if img is None:
        print(f"⚠️ No se pudo cargar {base}, se omite.")
        continue
    h, w = img.shape[:2]
    class_ids, polygons = leer_etiqueta_seg_yolo(ruta_lbl)

    # — Primero: guardar original —
    mask = polys_a_mask(polygons, class_ids, h, w)
    cv2.imwrite(os.path.join(carpeta_salida_imagenes, f"{base}_orig.jpg"), img)
    cv2.imwrite(os.path.join(carpeta_salida_etiquetas.replace("labels","masks"), f"{base}_orig.png"), mask)
    guardar_etiqueta_seg_yolo(
        os.path.join(carpeta_salida_etiquetas, f"{base}_orig.txt"),
        class_ids, polygons
    )

    # — Luego: por cada tipo de aumento —
    for name, aug in augmentations:
        transform = A.Compose([aug])
        augmented = transform(image=img, mask=mask)
        img_aug  = augmented["image"]
        mask_aug = augmented["mask"]

        cls_aug, polys_aug = mask_a_polys(mask_aug)

        # Guardar con sufijo del aumento
        cv2.imwrite(os.path.join(carpeta_salida_imagenes, f"{base}_aug_{name}.jpg"), img_aug)
        cv2.imwrite(os.path.join(carpeta_salida_etiquetas.replace("labels","masks"),
                                 f"{base}_aug_{name}.png"), mask_aug)
        guardar_etiqueta_seg_yolo(
            os.path.join(carpeta_salida_etiquetas, f"{base}_aug_{name}.txt"),
            cls_aug, polys_aug
        )

    print(f"✅ {base} procesado.")

print("✅ Procesamiento completado.")
