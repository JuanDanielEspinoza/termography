# # 🌡️ Thermal Image Processing for Regional Temperature Extraction

This repository contains tools and methods to automatically extract temperature values from specific body regions using **computer vision** and **thermal imaging (thermography)**.  
The goal of this project is to support medical research and clinical applications by enabling accurate, efficient, and automated analysis of thermal data.

---

## 📌 Objectives

- Automatically identify and segment relevant body regions in thermal images.  
- Extract temperature values from these regions (e.g., central and peripheral areas).  
- Provide a reproducible pipeline that can be integrated into larger research workflows.  
- Facilitate the creation of datasets for predictive modeling in clinical contexts.

---

## 🧠 Methodology

1. **Preprocessing**  
   - Image normalization and noise reduction.  
   - Optional background removal.

2. **Region Detection / Segmentation**  
   - Use of deep learning–based semantic segmentation or keypoint detection models.  
   - Manual annotation tools can be integrated during dataset creation.

3. **Temperature Extraction**  
   - Apply masks to the original thermal image.  
   - Compute temperature statistics (mean, max, min) for each region of interest.

4. **Data Export**  
   - Save extracted data in structured formats (e.g., CSV, JSON) for further analysis.

---

## 🧰 Tech Stack

- **Languages:** Python  
- **Libraries:** OpenCV, NumPy, Matplotlib, PyTorch, Albumentations
- **Eviroments:** uv
- **Annotation:** Label Studio 
- **Version Control:** Git + GitHub

---

## 📂 Repository Structure
```
thermal_temperature_extraction/
📂 data/                  # Thermal images and annotations (NOT versioned)
📂 notebooks/             # Jupyter notebooks for exploration & experiments
📂 src/
  📂 preprocessing/       # Image normalization, background removal, etc.
  📂 segmentation/        # Model definitions and inference scripts
  📂 augmentation/        # Albumentations pipelines
  📂 temperature_extraction/ # Temperature calculation scripts
📂 models/                # Model checkpoints (use Git LFS or .gitignore)
📂 outputs/               # Results, visualizations, extracted data
📄 requirements.txt       # Dependencies
📄 .gitignore
📄 README.md
```
---

## 👤 Authors
- **Juan Daniel Espinoza**
- **juan2248438@correo.uis.edu.co**
