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


# ------------------------------------------------------
# BIOMETRIC MEMORY (FACE + VOICE)
# ------------------------------------------------------

class BiometricMemory:
    """
    Stores and loads:
       • known_face_names      -> data/faces/face_db.json
       • known_voice_embeddings -> data/voice/voice_db.json
    """

    def __init__(self):
        # UPDATED PATHS
        self.face_db_path = config.FACE_DIR / "face_db.json"
        self.voice_db_path = config.VOICE_DIR / "voice_db.json"

        self.known_face_encodings = []
        self.known_face_names = []
        self.known_voice_embeddings = {}

        self.load_biometrics()

    # ------------------------
    # LOADING
    # ------------------------
    def load_biometrics(self):
        """ Load face encodings + voice embeddings from JSON files """

        # -------- FACES --------
        self.known_face_names = []
        self.known_face_encodings = []

        if self.face_db_path.exists():
            try:
                with open(self.face_db_path, "r") as f:
                    data = json.load(f)
                
                # Logic: Flatten the database so "face_recognition" library can use it
                for name, enc_list_of_lists in data.items():
                    # Check if it's the old format (single list) or new (list of lists)
                    if not enc_list_of_lists: continue
                    
                    # If it's a list of floats (Old format), wrap it
                    if isinstance(enc_list_of_lists[0], float):
                         self.known_face_names.append(name)
                         self.known_face_encodings.append(np.array(enc_list_of_lists))
                    else:
                        # NEW FORMAT: Multiple vectors per person
                        for enc in enc_list_of_lists:
                            self.known_face_names.append(name)
                            self.known_face_encodings.append(np.array(enc))
                            
                print(f"[MEMORY] Loaded {len(self.known_face_encodings)} Face Vectors.")
            except Exception as e:
                print(f"[MEMORY ERROR] Could not load face_db.json: {e}")

        # -------- VOICES --------
        self.known_voice_embeddings = {}

        if self.voice_db_path.exists():
            try:
                with open(self.voice_db_path, "r") as f:
                    data = json.load(f)
                for name, emb_list in data.items():
                    self.known_voice_embeddings[name] = np.array(emb_list)
                print(f"[MEMORY] Loaded {len(self.known_voice_embeddings)} Voice IDs from {self.voice_db_path}")
            except Exception as e:
                print(f"[MEMORY ERROR] Could not load voice_db.json: {e}")

    # ------------------------
    # SAVING (FLUSH TO DISK)
    # ------------------------
    def _flush_faces(self):
        # Group encodings by name -> {"Parth": [ [vector1], [vector2] ] }
        grouped_data = {}
        for name, enc in zip(self.known_face_names, self.known_face_encodings):
            if name not in grouped_data:
                grouped_data[name] = []
            grouped_data[name].append(enc.tolist())
            
        with open(self.face_db_path, "w") as f:
            json.dump(grouped_data, f, indent=2)

    def _flush_voices(self):
        serial = {n: emb.tolist() for n, emb in self.known_voice_embeddings.items()}
        with open(self.voice_db_path, "w") as f:
            json.dump(serial, f, indent=2)

    # ------------------------
    # NEW ENTRY SAVE
    # ------------------------
    def save_face(self, name, encoding):
        """ Appends a new face angle. Does NOT overwrite old ones. """
        self.known_face_names.append(name)
        self.known_face_encodings.append(encoding)
        self._flush_faces()
        print(f"[BIOMETRICS] New face angle saved for: {name}")

    def save_voice(self, name, embedding):
        """ Save or overwrite a voice embedding """
        self.known_voice_embeddings[name] = embedding
        self._flush_voices()
        print(f"[BIOMETRICS] Voice saved for: {name}")

    # ------------------------
    # REINFORCEMENT LEARNING (UPDATES)
    # ------------------------
    def reinforce_face(self, name, new_encoding):
        """
        Slightly adjusts the stored face encoding towards the new one.
        Weighted Average: 90% Old, 10% New.
        """
        if name in self.known_face_names:
            idx = self.known_face_names.index(name)
            old_enc = self.known_face_encodings[idx]
            
            # Math: Move 10% towards the new observation
            updated_enc = (old_enc * 0.9) + (new_encoding * 0.1)
            
            self.known_face_encodings[idx] = updated_enc
            self._flush_faces()
            # LOGGING ADDED HERE
            print(f"[LEARNING] Reinforced Face ID for {name}")

    def reinforce_voice(self, name, new_embedding):
        """
        Slightly adjusts the stored voice embedding.
        Weighted Average: 90% Old, 10% New.
        """
        if name in self.known_voice_embeddings:
            old_emb = self.known_voice_embeddings[name]
            updated_emb = (old_emb * 0.9) + (new_embedding * 0.1)
            
            # Re-normalize to keep it a valid unit vector
            norm = np.linalg.norm(updated_emb)
            if norm > 0:
                updated_emb = updated_emb / norm
            
            self.known_voice_embeddings[name] = updated_emb
            self._flush_voices()
            print(f"[LEARNING] Reinforced Voice ID for {name}")

    # ------------------------
    # IDENTIFICATION (MULTI-USER)
    # ------------------------
    def identify_all_faces(self, frame):
        """
        Scans entire frame.
        Returns a list of dictionaries:
        [
            {"name": "Parth", "encoding": [...], "center": (500, 300)},
            {"name": "Unknown", "encoding": [...], "center": (200, 300)}
        ]
        """

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        
        results = []

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                # Upscale coordinates
                center_x = int((left + right) / 2) * 4
                center_y = int((top + bottom) / 2) * 4
                
                name = "Unknown"
                
                # Compare against knowns
                if self.known_face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings,
                        encoding,
                        tolerance=0.6
                    )
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings,
                        encoding
                    )

                    if len(face_distances) > 0:
                        best_idx = np.argmin(face_distances)
                        if matches[best_idx]:
                            name = self.known_face_names[best_idx]
                
                results.append({
                    "name": name,
                    "encoding": encoding,
                    "center": (center_x, center_y)
                })

        return results


# ------------------------------------------------------
# TEXT MEMORY (FACTS / USER PROFILE)
# ------------------------------------------------------

class TextMemory:
    """ Stores facts per user. """

    def __init__(self):
        self.user_db_path = config.MEM_DIR / "user_profiles.json"
        self.user_data = {}
        self.load_memory()

    def load_memory(self):
        if self.user_db_path.exists():
            try:
                with open(self.user_db_path, "r") as f:
                    self.user_data = json.load(f)
            except:
                self.user_data = {}

    def save_memory(self):
        with open(self.user_db_path, "w") as f:
            json.dump(self.user_data, f, indent=2)

    def ensure_user_exists(self, name):
        """ Create profile if missing """
        if name not in self.user_data:
            self.user_data[name] = {"facts": []}
            self.save_memory()

    def add_user_fact(self, name, fact):
        """ Add a personal fact (only for known users) """
        if name == "Unknown":
            return
        self.ensure_user_exists(name)
        if fact not in self.user_data[name]["facts"]:
            self.user_data[name]["facts"].append(fact)
            self.save_memory()

    def get_user_context(self, name):
        """ Return formatted context string """
        if name not in self.user_data:
            return ""

        facts = self.user_data[name].get("facts", [])
        if not facts:
            return ""

        ctx = f"FACTS ABOUT {name}:\n"
        for f in facts:
            ctx += f"- {f}\n"
        return ctx.strip()