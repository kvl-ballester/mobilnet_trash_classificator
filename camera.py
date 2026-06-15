import cv2

class Camera:
    def __init__(self, resolution=(1280, 720)):
        self.resolution = resolution

        try:
            from picamera2 import Picamera2

            self.picam2 = Picamera2()

            config = self.picam2.create_preview_configuration(
                main={
                    "size": resolution,
                    "format": "BGR888"   # 👈 aquí lo que pedías
                }
            )

            self.picam2.configure(config)
            self.picam2.start()

            self.mode = "picamera"
            print("Usando Picamera2 (CSI - BGR888)")

        except Exception as e:
            print("Picamera2 no disponible, usando OpenCV:", e)

            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

            for _ in range(5):
                self.cap.read()

            self.mode = "cv2"
            print("Usando OpenCV")

    def read(self):
        if self.mode == "picamera":
            frame = self.picam2.capture_array()
            return frame  # ya debería venir en BGR

        ret, frame = self.cap.read()
        return frame if ret else None
