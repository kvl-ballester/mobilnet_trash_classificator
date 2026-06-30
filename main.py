import sys
from typing import Literal
import cv2
from time import perf_counter, sleep
from utils import preprocess_image, get_tflite, top_k_predictions, group_full
from collections import deque
import numpy as np
from camera import Camera
from model import TFLiteModel


BUFFER_LENGTH = 3
TOP_K = 5
CONFIDENCE = 0.95
ALLOWED_FAMILIES = ["carton", "plastic", "metal"]



MODEL_PATH = "models/mobilenetV3_trash_classifier/model.tflite"
CLASS_NAMES_PATH = "models/mobilenetV3_trash_classifier/class_names.txt"

MODEL_PATH = "models/mobilenetV3_trash_classifier_V2/model.tflite"
CLASS_NAMES_PATH = "models/mobilenetV3_trash_classifier_V2/class_names.txt"

DECODE_LEVEL: Literal["class", "garbage"] = "class"
CAMERA_RESOLUTION = (720, 720)


with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]
    class_to_family = {c: c.split("_")[0] for c in class_names if c.split("_")[0] in ALLOWED_FAMILIES}

# --- MODEL ---
tflite_model = TFLiteModel(MODEL_PATH, CLASS_NAMES_PATH, BUFFER_LENGTH)

# --- BUFFER ---
buffer = deque(maxlen=BUFFER_LENGTH)

# --- Camera ---
cam = Camera(resolution=CAMERA_RESOLUTION, use_opencv=True, show=True)


try:
    while True:
        frame = cam.read()
        if frame is None:
            break
        
        img = preprocess_image(frame, is_batch=False) #shape (224,224,3)
        buffer.append(img)
        if len(buffer) == buffer.maxlen:
            batch = np.stack(buffer, axis=0)  #(3,224,224,3)
            output = tflite_model.invoke(batch) #(3, 1, class_names)
            pred_mean = output.mean(axis=0, keepdims=True) #(1, 1, class_names)
            
            if DECODE_LEVEL == "class":
                predictions = tflite_model.get_top_predictions(pred_mean, 5)[0]
            if DECODE_LEVEL == "garbage":
                predictions = group_full(pred_mean[0], class_names, class_to_family)
            
            print(predictions)
            best_class = max(predictions, key=predictions.get)
            if predictions[best_class] >= CONFIDENCE and best_class != "person":
                print(f"{best_class} {predictions[best_class] * 100:.4}%")
                buffer.clear()
                input("Pulsar boton para continuar.")


except Exception as exc:
    print(exc)
