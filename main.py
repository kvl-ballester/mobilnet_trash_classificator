import sys
import cv2
from time import perf_counter, sleep
from utils import preprocess_image, get_tflite
from collections import deque, defaultdict
import numpy as np


BUFFER_LENGTH = 3
TOP_K = 5
CONFIDENCE = 0.95

# --- Load TFLite model ---
tflite = get_tflite()
interpreter = tflite.Interpreter(model_path="model/trash_classification_v3_float32.tflite")
input_details = interpreter.get_input_details()
interpreter.resize_tensor_input(
    input_details[0]["index"],
    [BUFFER_LENGTH, 224, 224, 3]
)

output_details = interpreter.get_output_details()
interpreter.allocate_tensors()

with open("model/class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
    class_to_family = {c: c.split("_")[0] for c in class_names}

# --- BUFFER ---
buffer = deque(maxlen=BUFFER_LENGTH)


cap = cv2.VideoCapture(0)
# warm-up cámara
for _ in range(5):
    cap.read()


def top_k_predictions(pred, class_names, k=3):
    idx = np.argsort(pred)[::-1][:k]
    return [(class_names[i], float(pred[i])) for i in idx]

def group_prediction(top_predictions):
    grouped = defaultdict(float)

    for label, prob in top_predictions:
        family = label.split("_")[0]
        grouped[family] += prob

    return grouped

def group_full(pred, class_names, class_to_family):
    grouped = defaultdict(float)

    for p, c in zip(pred, class_names):
        grouped[class_to_family[c]] += float(p)

    return grouped

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from webcam")
            break
        
        img = preprocess_image(frame, is_batch=False) #shape (224,224,3)
        buffer.append(img)
        if len(buffer) == buffer.maxlen:
            batch = np.stack(buffer, axis=0)  # (3,224,224,3)
            interpreter.set_tensor(input_details[0]["index"], batch)
            interpreter.invoke()

            output = interpreter.get_tensor(output_details[0]["index"])

            pred_mean = output.mean(axis=0)

            grouped_predictions = group_full(pred_mean, class_names, class_to_family)
            best_class = max(grouped_predictions, key=grouped_predictions.get)
            
            if grouped_predictions[best_class] >= CONFIDENCE:
                print(f"{best_class} {grouped_predictions[best_class] * 100:.4}%")
                buffer.clear()
                input("Pulsar boton para continuar.")



except Exception as exc:
    print(exc)