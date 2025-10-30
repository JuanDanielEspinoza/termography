import matplotlib.pyplot as plt

preds = model('/content/drive/MyDrive/train_pose/test/images_gray')
# Mostrar todas las predicciones al tiempo
cols = 3  # número de columnas que quieres mostrar
rows = (len(preds) + cols - 1) // cols

plt.figure(figsize=(15, 5 * rows))
for i, p in enumerate(preds):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(p.plot()[:, :, ::-1])  # convierte de BGR a RGB
    plt.axis('off')
    plt.title(f'Predicción {i+1}')
plt.tight_layout()
plt.show()