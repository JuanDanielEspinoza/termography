import cv2
import os

# Ruta de la imagen (ajusta según donde esté el archivo)
ruta_imagen = r"C:\Users\ASUS\Desktop\Canada\test_uv\can.png"

# Verificar si existe el archivo
if not os.path.exists(ruta_imagen):
    print(f"❌ Error: No se encuentra el archivo en: {ruta_imagen}")
    print(f"Por favor, verifica que 'can.png' esté en la carpeta correcta")
    exit()

# Cargar la imagen
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print("❌ Error: No se pudo cargar la imagen 'can.png'")
    exit()

print(f"📷 Tamaño original: {imagen.shape[1]}x{imagen.shape[0]}")

# Redimensionar a 160x120
imagen_resized = cv2.resize(imagen, (160, 120), interpolation=cv2.INTER_AREA)
print(f"📐 Redimensionado a: 160x120")

# Rotar 90 grados en sentido horario
imagen_rotada = cv2.rotate(imagen_resized, cv2.ROTATE_90_CLOCKWISE)
print(f"🔄 Rotado 90° en sentido horario")
print(f"📏 Tamaño final: {imagen_rotada.shape[1]}x{imagen_rotada.shape[0]}")

# Guardar la imagen procesada
cv2.imwrite(r"C:\Users\ASUS\Desktop\Canada\test_uv\can_procesada.png", imagen_rotada)
print(f"✅ Imagen guardada como: can_procesada.png")
