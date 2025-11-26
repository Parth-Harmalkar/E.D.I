import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Paths
BASE_DIR = Path("./data")
ASSETS_DIR = Path("./assets") 

MEM_DIR = BASE_DIR / "memory"
FACE_DIR = BASE_DIR / "faces"
VOICE_DIR = BASE_DIR / "voice"
TEMP_DIR = BASE_DIR / "temp"
MODELS_DIR = BASE_DIR / "models"

BRAIN_FILE = MEM_DIR / "brain.json"
FACE_DB_FILE = FACE_DIR / "face_db.json"
VOICE_DB_FILE = VOICE_DIR / "voice_db.json"
YOLO_BLOB = MODELS_DIR / "yolov8n_coco_640x352.blob"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"

# Camera
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# GUI
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 480
BG_COLOR = (10, 10, 20) 
EYE_COLOR = (0, 255, 255) 

# Behavior
AMBIENT_INTERVAL = 45.0

# --- NEW SETTING ---
# Set to False to hide the debug window and improve FPS
SHOW_CAMERA_FEED = True
YOLO_MODE = "FAST" # "HD" Or "FAST"