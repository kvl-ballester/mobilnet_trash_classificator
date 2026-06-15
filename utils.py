import cv2
import numpy as np
from typing import List, Tuple

TARGET_SIZE = 224
SCALE_SIZE = 256

def get_tflite():
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        from tensorflow import lite as tflite
    return tflite


tflite = get_tflite()

def is_raspberry_pi():
    try:
        with open("/proc/cpuinfo", "r") as f:
            return "raspberry" in f.read().lower()
    except Exception:
        return False

# --- Preprocessing like MobileNetV2 (tf mode) ---
def preprocess_image(frame: np.ndarray, is_batch: bool = True) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame to expected dimensions
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    # Pixel values between [-1, 1]
    frame_normalized = (frame_resized / 127.5 - 1).astype(np.float32)

    # Convert to tensor => (1, 224, 224, 3)
    if is_batch:
        return np.expand_dims(frame_normalized, axis=0)

    return frame_normalized


def print_inference_info(frame: np.ndarray, detection_str: str, preprocess_time: float, inference_time: float, input_shape: Tuple[int, ...]) -> None:
    """
    Prints formatted information about the processed frame.

    Args:
        frame (np.ndarray): Current frame image.
        detection_str (str): Detection string.
        preprocess_time (float): Preprocessing time in seconds.
        inference_time (float): Inference time in seconds.
        input_shape (tuple): Input tensor shape, e.g. (1, 3, 640, 640).
    """
    h, w = frame.shape[:2]
    print(f"\nimage 1/1: {h}x{w} {detection_str}")
    print(f"Speed: {preprocess_time * 1000:.1f}ms pre-process, {inference_time * 1000:.1f}ms inference per image at shape {input_shape}")


def get_top_predictions(output: np.ndarray, labels: List[str], n: int = 3) -> List[List[Tuple[str, float]]]:
    """
    Returns the top `n` predictions for each example in the batch as (label, score) pairs.

    Args:
        output (np.ndarray): 2D array (batch_size, num_classes) or 1D (num_classes,).
        labels (List[str]): List of class labels.
        n (int): Number of top predictions to return.

    Returns:
        List[List[Tuple[str, float]]]: List of lists of tuples (label, score), one list per image.
    """
    if output.ndim == 1:
        output = np.expand_dims(output, axis=0)  # convert (num_classes,) to (1, num_classes)

    result = []
    for row in output:
        top_indices = row.argsort()[-n:][::-1]
        top_preds = [(labels[i], float(row[i])) for i in top_indices]
        result.append(top_preds)

    return result


def group_probabilities_by_prefix(predictions):
    """
    Groups the probabilities of a multiclass classifier by the prefix of the class name.

    Args:
        predictions: A list of tuples, where each tuple contains the class name (string)
                      and its associated probability (float). Class names are expected to have a
                      common prefix followed by an underscore '_'.

    Returns:
        A dictionary where the keys are the class prefixes and the values are the sum
        of the probabilities corresponding to that prefix.
    """
    groups = {}
    for name, probability in predictions:
        prefix = name.split('_')[0]
        if prefix in groups:
            groups[prefix] += probability
        else:
            groups[prefix] = probability
    return groups

def get_grouped_prediction(top_predictions):
    """
    Returns the predicted superclass and its probability from grouped top predictions.

    Args:
        top_predictions (List[Tuple[str, float]]): Top predictions from the model.

    Returns:
        Tuple[str, float]: Predicted superclass and its grouped probability.
    """
    grouped_probs = group_probabilities_by_prefix(top_predictions)
    predicted_superclass = max(grouped_probs, key=grouped_probs.get)
    max_probability = grouped_probs[predicted_superclass]
    return predicted_superclass, max_probability
