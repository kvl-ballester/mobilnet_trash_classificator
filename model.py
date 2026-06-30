from utils import get_tflite
import numpy as np 

tflite = get_tflite()

class TFLiteModel():
    def __init__(self, model_path: str, class_names: str | list[str], batch_size: int = 1):

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.input_details = self.interpreter.get_input_details()
        try: 
            self.interpreter.resize_tensor_input(
                self.input_details[0]["index"],
                [batch_size, 224, 224, 3]
            )
        except Exception as e:
            print(f"Error al configurar la forma del tensor: {e}")
            raise
        
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        if isinstance(class_names, str):
            with open(class_names, "r") as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            self.class_names = class_names
        
        model_num_classes = self.output_details[0]["shape"][-1]
        if len(self.class_names) != model_num_classes:
            raise ValueError(
                f"Error de coincidencia: El modelo tiene {model_num_classes} salidas, "
                f"pero se han proporcionado {len(self.class_names)} etiquetas."
            )

    
    def invoke(self, batch: np.ndarray) -> np.ndarray:
        
        self.interpreter.set_tensor(self.input_details[0]["index"], batch)
        self.interpreter.invoke()

        return  self.interpreter.get_tensor(self.output_details[0]["index"])
    
    def get_top_predictions(self, output: np.ndarray, k: int = 1) -> list[dict]: 
    
        return [{self.class_names[i]: float(prediction[i]) for i in np.argsort(prediction)[::-1][:k]} for prediction in output]
        
        
    
    
    
