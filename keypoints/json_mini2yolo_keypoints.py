import json
import os

# Cargar tu archivo JSON-min
with open("project-2-at-2025-05-30-17-50-191835f7.json") as f:
    data = json.load(f)

# Carpeta donde se guardarán los .txt
output_dir = "labels"
os.makedirs(output_dir, exist_ok=True)

for task in data:
    keypoints = task["keypoints"]
    bboxes = task["bbox"]
    img_path = task["img"]
    img_filename = os.path.basename(img_path)
    txt_filename = os.path.splitext(img_filename)[0] + ".txt"
    output_path = os.path.join(output_dir, txt_filename)

    img_w = keypoints[0]["original_width"]
    img_h = keypoints[0]["original_height"]

    # Lista para acumular líneas YOLO pose por cada persona
    yolo_lines = []

    for bbox in bboxes:
        x = bbox["x"]
        y = bbox["y"]
        w = bbox["width"]
        h = bbox["height"]

        # Normalización bbox
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        line = [0, x_center, y_center, w_norm, h_norm]  # class-index = 0 (persona)

        # Keypoints dentro de esta caja
        for kp in keypoints:
            px = kp["x"]
            py = kp["y"]

            if x <= px <= x + w and y <= py <= y + h:
                px_norm = px / img_w
                py_norm = py / img_h
                visibility = 2  # Ajusta si tienes ese dato
                line.extend([px_norm, py_norm, visibility])

        if len(line) > 5:  # Asegurar que haya al menos un keypoint
            yolo_lines.append(" ".join(map(str, line)))

    # Escribir archivo por imagen
    with open(output_path, "w") as f:
        for line in yolo_lines:
            f.write(line + "\n")

