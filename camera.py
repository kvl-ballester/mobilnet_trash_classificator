import cv2

class Camera:
    def __init__(self, resolution=(1280, 720), use_opencv=False):
        self.resolution = resolution

        if not use_opencv:
            try:
                from picamera2 import Picamera2

                self.picam2 = Picamera2()

                config = self.picam2.create_preview_configuration(
                    main={
                        "size": resolution,
                        "format": "BGR888"
                    }
                )

                self.picam2.configure(config)
                self.picam2.start()

                self.mode = "picamera"
                print("Usando Picamera2 (CSI - BGR888)")
                return

            except Exception as e:
                print("Picamera2 no disponible, usando OpenCV:", e)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        for _ in range(5):
            self.cap.read()

        self.mode = "cv2"
        print("Usando OpenCV")

    def read(self, show: bool = False):
        frame = None
        if self.mode == "picamera":
            frame = self.picam2.capture_array()
            
        if self.mode == "cv2":
            ret, frame = self.cap.read()
            if not ret:
                return None
        
        if show:
            cv2.imshow("Camera", frame)
            cv2.waitKey(1)
            
        return frame
