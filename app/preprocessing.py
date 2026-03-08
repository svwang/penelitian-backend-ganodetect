# preprocessing.py
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import resnet50, efficientnet, mobilenet_v2
import logging

logger = logging.getLogger(__name__)

IMG_SIZE = (224, 224)

def load_image(file_bytes):
    """
    Load image dari bytes → RGB → resize → float32
    """
    logger.info("Converting image to RGB...")
    img = Image.open(BytesIO(file_bytes)).convert("RGB")

    logger.info(f"Resizing image to {IMG_SIZE}...")
    img = img.resize(IMG_SIZE)

    logger.info("Converting image to float32 array...")
    img = np.array(img, dtype=np.float32)

    logger.info("Adding batch dimension...")
    img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)
    return img


def preprocess_resnet(img):
    return resnet50.preprocess_input(img)


def preprocess_efficientnet(img):
    return efficientnet.preprocess_input(img)


def preprocess_mobilenet(img):
    return mobilenet_v2.preprocess_input(img)
