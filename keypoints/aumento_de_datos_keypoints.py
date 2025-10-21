#Este codigo soluciona la perdida de puntos a la hora de realizar el aumento de datos manteniendo
#las 56 columnas pertinentes para el formato yolo
import cv2
import albumentations as A
import os
from glob import glob
import numpy as np

def leer_etiqueta_pose_yolo(ruta_txt):
    with open(ruta_txt, 'r') as f:
        contenido = f.read().strip()
    if not contenido:
        raise ValueError(f"El archivo {ruta_txt} está vacío.")
    valores = list(map(float, contenido.split()))
    clase = int(valores[0])
    bbox = valores[1:5]
    keypoints = []
    for i in range(5, len(valores), 3):
        if i + 2 < len(valores):
            x = valores[i]
            y = valores[i + 1]
            v = valores[i + 2]
            keypoints.append((x, y, v))
    return clase, bbox, keypoints

def guardar_etiqueta_yolo(ruta_salida, clase, bbox, keypoints):
    linea = [str(clase)]
    linea += [f"{x:.6f}" for x in bbox]
    for x, y, v in keypoints:
        linea += [f"{x:.6f}", f"{y:.6f}", str(int(v))]    
    contenido = " ".join(linea)
    with open(ruta_salida, "w") as f:
        f.write(contenido)

# === RUTAS DE LAS CARPETAS ===
carpeta_imagenes = r"E:\descargas\Train_yolo\train_pose\test\images"
carpeta_etiquetas = r"E:\descargas\Train_yolo\train_pose\test\labels"
carpeta_salida_imagenes = r"E:\descargas\Train_yolo\key_points_aumented\test\images"
carpeta_salida_etiquetas = r"E:\descargas\Train_yolo\key_points_aumented\test\labels"

os.makedirs(carpeta_salida_imagenes, exist_ok=True)
os.makedirs(carpeta_salida_etiquetas, exist_ok=True)

lab_clases = [
    'px1-Nariz',
    'px2-Ojo izquierdo',
    'px3-Ojo derecho',
    'px4-Oreja izquierda',
    'px5-Oído derecho',
    'px6-Hombro izquierdo',
    'px7-Hombro derecho',
    'px8-Codo izquierdo',
    'px9-Codo derecho',
    'px10-Muñeca izquierda',
    'px11-Muñeca derecha',
    'px12-Cadera izquierda',
    'px13-Cadera derecha',
    'px14-Rodilla izquierda',
    'px15-Rodilla derecha',
    'px16-Tobillo izquierdo',
    'px17-Tobillo derecho',
]

# === DEFINIR TRANSFORMACIONES ===
keypoint_params = A.KeypointParams(
    format='xy',
    remove_invisible=False,
    label_fields=["keypoint_labels"]
)

bbox_params = A.BboxParams(
    format='yolo',
    label_fields=["bbox_labels"]
)

# Lista de transformaciones
lista_transforms = [
    A.Compose([
        A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, p=1.0),
        A.RandomBrightnessContrast(p=1.0)
    ], keypoint_params=keypoint_params, bbox_params=bbox_params),

    A.Compose([
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(p=1.0),
    ], keypoint_params=keypoint_params, bbox_params=bbox_params),

    A.Compose([
        A.GridDistortion(p=1.0)
    ], keypoint_params=keypoint_params, bbox_params=bbox_params),

    A.Compose([
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-30, 30),
            shear={"x": (-10, 10), "y": (-5, 5)},
            p=1.0
        ),
        A.MotionBlur(p=1.0)
    ], keypoint_params=keypoint_params, bbox_params=bbox_params),

    A.Compose([
        A.Perspective(scale=(0.05, 0.1), keep_size=True, p=1.0)
    ], keypoint_params=keypoint_params, bbox_params=bbox_params)
]

# === PROCESAMIENTO ===
rutas_imagenes = glob(os.path.join(carpeta_imagenes, "*.jpg"))

for ruta_imagen in rutas_imagenes:
    nombre_archivo = os.path.basename(ruta_imagen)
    nombre_base, _ = os.path.splitext(nombre_archivo)
    ruta_etiqueta = os.path.join(carpeta_etiquetas, nombre_base + ".txt")

    if not os.path.exists(ruta_etiqueta):
        print(f"⚠️ No se encontró etiqueta para {nombre_archivo}, se omite.")
        continue

    img = cv2.imread(ruta_imagen)
    if img is None:
        print(f"⚠️ No se pudo cargar {ruta_imagen}, se omite.")
        continue

    h1, w1 = img.shape[:2]
    clase, bbox, keypoints = leer_etiqueta_pose_yolo(ruta_etiqueta)

    keypoints_abs = []
    vis_list = []
    for x, y, v in keypoints:
        x_abs = x * w1 if x <= 1 else x
        y_abs = y * h1 if y <= 1 else y
        keypoints_abs.append((x_abs, y_abs))
        vis_list.append(v)
        print("visibilidad: ",vis_list)

    keypoint_labels = lab_clases[:len(keypoints_abs)]
    bbox_labels = [clase]

    # Guardar imagen original
    ruta_img_orig = os.path.join(carpeta_salida_imagenes, f"{nombre_base}_orig.jpg")
    ruta_lbl_orig = os.path.join(carpeta_salida_etiquetas, f"{nombre_base}_orig.txt")
    cv2.imwrite(ruta_img_orig, img)
    guardar_etiqueta_yolo(ruta_lbl_orig, clase, bbox, keypoints)

    for idx, transform in enumerate(lista_transforms, start=1):
        try:
            aug = transform(
                image=img,
                keypoints=keypoints_abs,
                keypoint_labels=keypoint_labels,
                bboxes=[bbox],
                bbox_labels=bbox_labels
            )
        except Exception as e:
            print(f"⚠️ Error en transformación {idx} de {nombre_base}: {e}")
            continue

        img_aug = aug['image']
        bbox_aug = aug['bboxes'][0]
        keypoints_aug = aug['keypoints']
        labels_aug = aug['keypoint_labels']
        print("transformed_class_labels ",len(aug['keypoint_labels']))
        print(keypoints_aug)
        if len(keypoints_aug) != 17:
            presentes = []
            for label in labels_aug:
                num = int(label.split('-')[0].replace('px', ''))
                presentes.append(num)

            todos = set(range(1, 18))
            faltan = sorted(list(todos - set(presentes)))

            print("Números presentes:", presentes)
            print("Números faltantes:", faltan)

            # Crear un diccionario num_label -> keypoint original (respetando visibilidad)
            num_to_kpt = {}
            for label, kpt in zip(labels_aug, keypoints_aug):
                num = int(label.split('-')[0].replace('px', ''))
                num_to_kpt[num] = kpt  # (x, y, v)

    # Reconstruir lista completa del 1 al 17
            keypoints_completos = []
            for i in range(1, 18):
                if i in num_to_kpt:
                    keypoints_completos.append(num_to_kpt[i])
                else:
                    keypoints_completos.append([0.0, 0.0])

            print("✅ Keypoints reconstruidos con placeholders:")
            print(keypoints_completos)
        else:
            keypoints_completos = keypoints_aug

        h2, w2 = img_aug.shape[:2]

        keypoints_norm = []
        visibles = []
        for i, (x, y) in enumerate(keypoints_completos):
            x_norm = x / w2
            y_norm = y / h2

            x_norm = max(0.0, min(1.0, x_norm))  # Asegura 0 ≤ x_norm ≤ 1
            y_norm = max(0.0, min(1.0, y_norm)) 
            v_orig = vis_list[i]
            v_new = v_orig if 0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0 else 0
            keypoints_norm.append((x_norm, y_norm, v_new))

            if v_new != 0:
                visibles.append(keypoints_completos[i])
            print("bandera: ",keypoints_norm)

        ruta_imagen_salida = os.path.join(carpeta_salida_imagenes, f"{nombre_base}_aug{idx}.jpg")
        ruta_etiqueta_salida = os.path.join(carpeta_salida_etiquetas, f"{nombre_base}_aug{idx}.txt")

        cv2.imwrite(ruta_imagen_salida, img_aug)
        guardar_etiqueta_yolo(ruta_etiqueta_salida, clase, bbox_aug, keypoints_norm)

        print(f"✅ Procesado {nombre_base} transformación {idx}")

print("✅ Procesamiento completado.")
