import cv2
import os
from glob import glob
import matplotlib.pyplot as plt

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
        if i + 1 < len(valores):
            x = valores[i]
            y = valores[i + 1]
            keypoints.append((x, y))
    return clase, bbox, keypoints

def mostrar_imagen_keypoints(img, keypoints, bbox=None, color=(0, 255, 0)):
    img_out = img.copy()
    h, w = img.shape[:2]
    for i, (x, y) in enumerate(keypoints):
        cx, cy = int(x * w), int(y * h)
        cv2.circle(img_out, (cx, cy), 4, color, -1)
        cv2.putText(img_out, f"{i}", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    if bbox:
        x_c, y_c, bw, bh = bbox
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# === RUTAS DE LAS CARPETAS ===
carpeta_imagenes = r"E:\descargas\Train_yolo\key_points_aumented\train\images2"
carpeta_etiquetas = r"E:\descargas\Train_yolo\key_points_aumented\train\labels2"


# carpeta_imagenes = r"E:\descargas\Train_yolo\key_points_aumented\test\images"
# carpeta_etiquetas = r"E:\descargas\Train_yolo\key_points_aumented\test\labels"


# === LISTAR TODAS LAS IMÁGENES ===
rutas_imagenes = glob(os.path.join(carpeta_imagenes, "*.jpg"))

# === PROCESAR CADA IMAGEN ===
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

    h, w = img.shape[:2]

    clase, bbox, keypoints = leer_etiqueta_pose_yolo(ruta_etiqueta)

    # No hace falta convertir keypoints: se asume que están normalizados
    mostrar_imagen_keypoints(img, keypoints, bbox)

print("✅ Todas las imágenes mostradas.")
