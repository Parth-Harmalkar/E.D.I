import cv2
import mediapipe as mp
import math
import time
import threading
from collections import deque, Counter
from deepface import DeepFace
import warnings

import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="keras")


import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)



"""
=========================================================
      VISION ENGINE (ULTRA-LIGHT DEEPFACE + OPTIMIZED)
=========================================================
"""

class VisionSystem:
    def __init__(self):
        print("[SYSTEM] Initializing Vision Engine (MediaPipe + DeepFace)...")
        
        # 1. SETUP MEDIAPIPE FACE MESH
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,  # Limit to 2 faces
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 2. SETUP MEDIAPIPE HANDS
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Drawing Utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.mesh_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
        self.contour_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(255, 255, 255))

        # 3. EMOTION (ULTRA-LIGHT MODE)
        self.emotion_history = {} 
        self.last_emotion_scan = 0
        self.EMOTION_INTERVAL = 22.0  # Run emotion every 22 seconds (super lightweight)
        self.is_processing_emotion = False  # Lock to avoid thread stacking

    # ---------------------------------------------------------
    # BASIC LIP MOVEMENT (Independent of DeepFace)
    # ---------------------------------------------------------
    def _get_mouth_height(self, landmarks, frame_h, frame_w):
        upper = landmarks[13]
        lower = landmarks[14]
        dist = math.hypot(
            (upper.x - lower.x) * frame_w, 
            (upper.y - lower.y) * frame_h
        )
        return dist

    # ---------------------------------------------------------
    # ASYNC EMOTION DETECTOR (Ultra-Light)
    # ---------------------------------------------------------
    def _detect_emotion_async(self, face_img, name):
        """Run DeepFace in the background without lag."""

        try:
            # Resize to tiny 120×120 for FAST deepface inference
            face_img = cv2.resize(face_img, (120, 120))

            objs = DeepFace.analyze(
                img_path=face_img,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )

            if objs:
                dom = objs[0]['dominant_emotion']

                if name not in self.emotion_history:
                    self.emotion_history[name] = deque(maxlen=5)

                self.emotion_history[name].append(dom)

        except:
            pass

        finally:
            self.is_processing_emotion = False

    # ---------------------------------------------------------
    # EMOTION STABILIZATION
    # ---------------------------------------------------------
    def _get_stable_emotion(self, name):
        if name not in self.emotion_history or not self.emotion_history[name]:
            return "neutral"
        counts = Counter(self.emotion_history[name])
        return counts.most_common(1)[0][0]

    # ---------------------------------------------------------
    # MAIN ANALYSIS PIPELINE
    # ---------------------------------------------------------
    def analyze(self, frame, known_faces):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = []
        
        # -----------------------------------------------------
        # A. FACE MESH + LIP MOTION
        # -----------------------------------------------------
        mesh_results = self.face_mesh.process(rgb_frame)

        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:

                # Draw Mesh
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mesh_spec
                )
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.contour_spec
                )

                # 1. Find person identity (match by proximity to face_recognition center)
                nose_tip = face_landmarks.landmark[1]
                cx, cy = int(nose_tip.x * w), int(nose_tip.y * h)
                
                matched_name = "Unknown"
                for kf in known_faces:
                    kx, ky = kf["center"]
                    dist = math.hypot(cx - kx, cy - ky)
                    if dist < 80:
                        matched_name = kf["name"]

                # 2. Lip Movement Detection
                mouth_height = self._get_mouth_height(face_landmarks.landmark, h, w)
                is_talking = mouth_height > 6.0

                # -----------------------------------------------------
                # EMOTION (Only run if enough time passed)
                # -----------------------------------------------------
                emotion = self._get_stable_emotion(matched_name)

                # Prepare small ROI
                y1, y2 = max(0, cy - 80), min(h, cy + 80)
                x1, x2 = max(0, cx - 80), min(w, cx + 80)
                face_roi = frame[y1:y2, x1:x2]

                # Trigger DeepFace if:
                # - 22s passed
                # - No other thread running
                # - Face ROI valid
                # - Not Unknown
                if (
                    time.time() - self.last_emotion_scan > self.EMOTION_INTERVAL
                    and not self.is_processing_emotion
                    and matched_name != "Unknown"
                    and face_roi.size > 0
                ):
                    self.is_processing_emotion = True
                    self.last_emotion_scan = time.time()

                    threading.Thread(
                        target=self._detect_emotion_async,
                        args=(face_roi.copy(), matched_name),
                        daemon=True
                    ).start()

                results.append({
                    "name": matched_name,
                    "center": (cx, cy),
                    "is_talking": is_talking,
                    "emotion": emotion or "neutral",
                    "gestures": []
                })

        # -----------------------------------------------------
        # B. HAND GESTURES
        # -----------------------------------------------------
        hand_results = self.hands.process(rgb_frame)
        detected_gestures = []

        if hand_results.multi_hand_landmarks:
            for hl in hand_results.multi_hand_landmarks:

                # Draw Hand Skeleton
                self.mp_drawing.draw_landmarks(
                    frame,
                    hl,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

                thumb_tip = hl.landmark[4]
                index_tip = hl.landmark[8]
                wrist = hl.landmark[0]

                # Gesture: 👍
                if thumb_tip.y < index_tip.y and index_tip.y < wrist.y:
                    detected_gestures.append("Thumbs Up")

                # Gesture: Waving (4 fingers up)
                fingers_up = 0
                if hl.landmark[8].y < hl.landmark[6].y: fingers_up += 1
                if hl.landmark[12].y < hl.landmark[10].y: fingers_up += 1
                if hl.landmark[16].y < hl.landmark[14].y: fingers_up += 1
                if hl.landmark[20].y < hl.landmark[18].y: fingers_up += 1

                if fingers_up >= 4:
                    detected_gestures.append("Waving")

        if results and detected_gestures:
            results[0]["gestures"] = list(set(detected_gestures))

        return results
