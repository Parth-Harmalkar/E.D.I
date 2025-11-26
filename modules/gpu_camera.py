from ultralytics import YOLO
import cv2
from modules import config

class GPUDetector:
    def __init__(self):
        print("[GPU YOLO] Initializing...")
        self.model = YOLO(str(config.YOLO_MODEL_PATH))      # PT model, not blob
        self.model.to("cuda")                # Force GPU

    def detect(self, frame):
        """
        Returns (raw_dets, formatted_labels)
        raw_dets = [{
            "bbox": (x1,y1,x2,y2),
            "class_name": "...",
            "confidence": 0.89
        }]
        formatted_labels = ["cup (0.89)", "person (0.76)"]
        """
        # Resize frame if FAST mode
        original_frame = frame

        if config.YOLO_MODE == "FAST":
            frame = cv2.resize(frame, (640, 352))

        results = self.model(frame, imgsz=640, verbose=False)
        raw_dets = []
        formatted = []

        # YOLO parsing
        r = results[0]
        for b in r.boxes:
            cls = int(b.cls[0])
            name = self.model.names[cls]
            conf = float(b.conf[0])
            x1, y1, x2, y2 = b.xyxy[0].tolist()

            if config.YOLO_MODE == "FAST":
                # scale coords back to original size
                scale_x = original_frame.shape[1] / 640
                scale_y = original_frame.shape[0] / 352
                x1 *= scale_x; x2 *= scale_x
                y1 *= scale_y; y2 *= scale_y

            raw_dets.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "class_name": name,
                "confidence": conf
            })

            formatted.append(f"{name} ({conf:.2f})")

        return raw_dets, formatted
