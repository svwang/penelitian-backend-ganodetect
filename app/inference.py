# inference.py
import tensorflow as tf
import numpy as np
import time
import os
import cv2

from .gradcam import make_gradcam_heatmap, overlay_heatmap
from .preprocessing import (
    load_image,
    preprocess_resnet,
    preprocess_efficientnet,
    preprocess_mobilenet
)

# =========================
# Path model
# =========================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

# =========================
# Load semua model (sekali saja)
# =========================
resnet_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "model_ganodetect_resnet50.keras")
)

efficientnet_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "model_ganodetect_efficientnet.keras")
)

mobilenet_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "model_ganodetect_mobilenet.keras")
)

print("✅ Semua model ensemble berhasil dimuat")

# =========================
# Konfigurasi
# =========================
CLASS_NAMES = ["Ganoderma", "Sehat"]

WEIGHTS = {
    "resnet": 0.30,
    "efficient": 0.45,
    "mobilenet": 0.25
}

# =========================
# Fungsi inference (WHITE BOX VERSION)
# =========================
def predict(file_bytes):

    process_steps = []
    start_time = time.time()

    process_steps.append("1️⃣ Gambar diterima oleh sistem")

    # =========================
    # Load image
    # =========================
    img = load_image(file_bytes)
    process_steps.append(f"2️⃣ Gambar di-resize menjadi 224x224")
    process_steps.append(f"   Shape setelah load: {img.shape}")

    # =========================
    # Preprocessing per model
    # =========================
    img_resnet = preprocess_resnet(img.copy())
    process_steps.append("3️⃣ Preprocessing ResNet dilakukan")

    img_effnet = preprocess_efficientnet(img.copy())
    process_steps.append("4️⃣ Preprocessing EfficientNet dilakukan")

    img_mobilenet = preprocess_mobilenet(img.copy())
    process_steps.append("5️⃣ Preprocessing MobileNet dilakukan")

    # =========================
    # Predict per model
    # =========================
    resnet_pred = resnet_model.predict(img_resnet, verbose=0)
    process_steps.append(f"6️⃣ Output ResNet: {resnet_pred.tolist()}")

    efficient_pred = efficientnet_model.predict(img_effnet, verbose=0)
    process_steps.append(f"7️⃣ Output EfficientNet: {efficient_pred.tolist()}")

    mobilenet_pred = mobilenet_model.predict(img_mobilenet, verbose=0)
    process_steps.append(f"8️⃣ Output MobileNet: {mobilenet_pred.tolist()}")

    # =========================
    # Ensemble berbobot
    # =========================
    ensemble_pred = (
        WEIGHTS["resnet"] * resnet_pred +
        WEIGHTS["efficient"] * efficient_pred +
        WEIGHTS["mobilenet"] * mobilenet_pred
    )

    process_steps.append(f"9️⃣ Hasil Weighted Ensemble: {ensemble_pred.tolist()}")

    class_idx = int(np.argmax(ensemble_pred))
    confidence = float(ensemble_pred[0][class_idx])
    label = CLASS_NAMES[class_idx]

    process_steps.append("🔟 Probabilitas akhir per kelas:")

    for i, prob in enumerate(ensemble_pred[0]):
        process_steps.append(
            f"   ➜ {CLASS_NAMES[i]}: {round(prob * 100, 2)}%"
        )

    process_steps.append(
        f"1️⃣1️⃣ Prediksi akhir: {label} ({round(confidence * 100, 2)}%)"
    )

    # =========================
    # Grad-CAM (EfficientNet)
    # =========================
    LAST_CONV_LAYER = "top_conv"

    heatmap = make_gradcam_heatmap(
        img_effnet,
        efficientnet_model,
        LAST_CONV_LAYER,
        pred_index=class_idx
    )

    original_img = (img[0]).astype("uint8")
    gradcam_image = overlay_heatmap(heatmap, original_img)

    gradcam_filename = f"gradcam_{int(time.time())}.jpg"
    cv2.imwrite(gradcam_filename, gradcam_image)

    process_steps.append("1️⃣2️⃣ Grad-CAM dibuat untuk visualisasi area fokus model")

    # =========================
    # Waktu inferensi
    # =========================
    end_time = time.time()
    inference_time = round(end_time - start_time, 3)

    process_steps.append(f"1️⃣3️⃣ Waktu inferensi: {inference_time} detik")

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "inference_time": inference_time,
        "whitebox_process": process_steps,
        "gradcam_image": gradcam_filename
    }