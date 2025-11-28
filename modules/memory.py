import json
import numpy as np
import face_recognition
from pathlib import Path
import threading
import time
import os
import cv2
from . import config

"""
=========================================================
  MEMORY SYSTEM (SEPARATE FACE & VOICE STORAGE)
  + EXPLICIT REINFORCEMENT LOGGING
=========================================================
"""

class DataManager:
    def __init__(self):
        # Ensure all data directories exist
        for d in [config.MEM_DIR, config.FACE_DIR, config.VOICE_DIR, config.TEMP_DIR, config.MODELS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def get_temp_file(self):
        """ Temporary WAV file for TTS output """
        return str(config.TEMP_DIR / f"temp_{threading.get_ident()}_{int(time.time())}.wav")

    def clear_temp(self):
        """ Remove all temporary audio files """
        for f in config.TEMP_DIR.glob("*"):
            try:
                f.unlink()
            except:
                pass


DM = DataManager()



