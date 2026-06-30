import os
import cv2
import numpy as np
from tqdm import tqdm
from model import TFLiteModel
from utils import preprocess_image
from pathlib import Path


def draw_predictions(img, predictions, title, x):
    font_scale: float = 0.5
    thickness: int = 1
    line_separation: int = 20
    cv2.putText(
        img,
        title,
        (x, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    y = 70

    predictions = sorted(
        predictions.items(),
        key=lambda item: item[1],
        reverse=True
    )

    for cls, prob in predictions:

        cv2.putText(
            img,
            f"{cls}: {prob:.3f}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            thickness,
        )

        y += line_separation
        
def get_output_path(input_path):
    base_output_dir = Path("./runs/compare")

    input_path = Path(input_path)
    filename = input_path.stem
    extension = input_path.suffix.lower()

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

    # Carpeta con el nombre del archivo de entrada
    output_dir = base_output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)

    if extension in image_extensions:
        return str(output_dir / f"result{extension}")
    elif extension in video_extensions:
        return str(output_dir / "result.mp4")
    else:
        raise ValueError(f"Formato no soportado: {extension}")
        
def process_frame(frame, model_1: TFLiteModel, model_2: TFLiteModel):

    input = preprocess_image(frame)

    output1 = model_1.invoke(input)
    pred1 = model_1.get_top_predictions(output1, 5)[0]

    output2 = model_2.invoke(input)
    pred2 = model_2.get_top_predictions(output2, 5)[0]

    result = frame.copy()

    h, w = result.shape[:2]

    # Modelo 1 a la izquierda
    draw_predictions(result, pred1, "MODEL 1", x=10)

    # Modelo 2 a la derecha
    draw_predictions(result, pred2, "MODEL 2", x=w // 2 + 10)

    return result

# ==========================================================
# Imagen
# ==========================================================

def process_image(input_path, model_1: TFLiteModel, model_2: TFLiteModel):

    output_path = get_output_path(input_path)
    
    image = cv2.imread(input_path)

    result = process_frame(image, model_1, model_2)

    cv2.imwrite(output_path, result)

    print(f"Guardado en {output_path}")


# ==========================================================
# Video
# ==========================================================

def process_video(input_path, model_1: TFLiteModel, model_2: TFLiteModel):
    output_path = get_output_path(input_path)
    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    with tqdm(total=total_frames, desc="Procesando vídeo") as pbar:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            result = process_frame(frame, model_1, model_2)

            writer.write(result)

            pbar.update(1)

    cap.release()
    writer.release()

    print(f"Guardado en {output_path}")



if __name__ == "__main__":

    input_file = "files\metal_soda_20260619_102147.mp4"
    
    model_path = "models/mobilenetV3_trash_classifier/model.tflite"
    class_names = "models/mobilenetV3_trash_classifier/class_names.txt"

    model_v2_path = "models/mobilenetV3_trash_classifier_V2/model.tflite"
    class_names_v2 = "models/mobilenetV3_trash_classifier_V2/class_names.txt"
    
    model_v1 = TFLiteModel(model_path, class_names)
    model_v2 = TFLiteModel(model_v2_path, class_names_v2)
    

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    ext = os.path.splitext(input_file)[1].lower()

    if ext in image_extensions:

        process_image(input_file, model_v1, model_v2)

    else:

        process_video(input_file, model_v1, model_v2)