import nltk
import blobconverter
import sys
import shutil
from . import config

def ensure_utils():
    needed = ['averaged_perceptron_tagger_eng', 'cmudict', 'punkt_tab']
    for n in needed:
        try: nltk.data.find(f'taggers/{n}')
        except: 
            try: nltk.data.find(f'corpora/{n}')
            except: nltk.download(n, quiet=True)

    if not config.YOLO_BLOB.exists():
        try:
            print("[INSTALL] Downloading YOLOv8n model...")
            downloaded_path = blobconverter.from_zoo(
                name="yolov8n_coco_640x352", 
                zoo_type="depthai", 
                shaves=6 
            )
            shutil.move(str(downloaded_path), str(config.YOLO_BLOB))
            print(f"[INSTALL] Model saved to: {config.YOLO_BLOB}")
            
        except Exception as e:
            print(f"[CRITICAL] Model download failed: {e}")
            sys.exit(1)