# --- 1. SILENCE WARNINGS (MUST BE AT TOP) ---
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import warnings
warnings.filterwarnings("ignore")

import functools
import re
import random
import threading
import time
import pygame
import cv2
import numpy as np
import face_recognition

# --- 2. IMPORTS ---
from modules import config
from modules.gpu_camera import GPUDetector  # NEW GPU YOLO REPLACEMENT
from modules.voice import VoiceAssistant
from modules.brain import GeminiBrain
from modules.gui import FridayUI
from modules.memory import DM
from modules.voice_auth import VoiceAuth
from modules.vision import VisionSystem

print = functools.partial(print, flush=True)

# --------------------------------------------------------------
# GLOBAL STATE
# --------------------------------------------------------------
current_active_user = "Unknown"
interaction_id = 0
onboarding_queue = [] 

# --------------------------------------------------------------
# DEBUG HUD (UPDATED TO GPU YOLO)
# --------------------------------------------------------------
def draw_debug_hud(frame, raw_dets, labels, vision_context=None):
    if not config.SHOW_CAMERA_FEED: 
        return

    height, width = frame.shape[:2]

    # 1. Draw YOLO GPU Objects
    if raw_dets:
        for d in raw_dets:
            try:
                x1, y1, x2, y2 = d["bbox"]
                name = d["class_name"]
                conf = d["confidence"]

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            except:
                pass

    # 2. Draw Vision Context (Lips/Emotions)
    if vision_context:
        for p in vision_context:
            cx, cy = p['center']
            color = (0, 255, 255) if p['is_talking'] else (0, 0, 255)
            status = "SPEAKING" if p['is_talking'] else "Silent"
            text = f"{p['name']} | {p['emotion']} | {status}"
            if p['gestures']:
                text += f" | {','.join(p['gestures'])}"

            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, text, (cx+10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("FRIDAY Social Vision", frame)
    cv2.waitKey(1)
# --------------------------------------------------------------
#  ROBUST ONBOARDING WIZARD (More Data, Better Voice)
#  (UPDATED: Removed OAK queues, now uses webcam frames)
# --------------------------------------------------------------
def run_onboarding_wizard(name, camera, voice, brain, gui, voice_auth):
    print(f"\n[ONBOARDING] INITIALIZING PROTOCOL FOR: {name}")
    gui.set_emotion("HAPPY")
    
    voice.speak(f"Starting biometric calibration for {name}. Please get ready.")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"--> {i}")
        time.sleep(1.0)

    # We capture multiple samples per angle to ensure quality
    angles = [
        ("Look straight ahead", 2.0),
        ("Turn head LEFT", 2.0),
        ("Turn head RIGHT", 2.0),
        ("Tilt head UP", 2.0),
        ("Tilt head DOWN", 2.0)
    ]

    captured_count = 0
    
    for instruction, duration in angles:
        voice.speak(instruction)
        print(f"\n[WIZARD] {instruction}...")
        
        start_t = time.time()
        samples_this_angle = 0
        
        while time.time() - start_t < duration:
            ret, frame = camera.read()
            if not ret:
                continue

            # Feedback
            cv2.putText(frame, f"HOLD: {instruction}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("FRIDAY Social Vision", frame)
            cv2.waitKey(1)

            # Capture up to 2 frames per angle (Redundancy)
            if samples_this_angle < 2:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb)
                
                if len(boxes) == 1:
                    enc = face_recognition.face_encodings(rgb, boxes)[0]
                    brain.bio_mem.save_face(name, enc)
                    print(f"[WIZARD] Captured Sample {samples_this_angle+1} for: {instruction}")
                    samples_this_angle += 1
                    captured_count += 1
                    time.sleep(0.3) # Space them out slightly

    if captured_count < 5:
        voice.speak("I missed some angles, but I saved what I could.")
    else:
        voice.speak(f"I have captured {captured_count} biometric reference points.")

    # --- VOICE CALIBRATION ---
    phrase = "I authorize the Friday system to capture my biometric data."
    
    attempts = 0
    while attempts < 3:
        attempts += 1
        print(f"\n[WIZARD] PLEASE READ: \"{phrase}\"")
        voice.speak("Now for voice. Please read the sentence on screen.")
        time.sleep(4.5)
        
        print("[BEEP] LISTENING... (Speak Clearly)")
        
        voice_emb = voice_auth.get_embedding_from_mic(duration=6.0)

        if voice_emb is not None:
            brain.bio_mem.save_voice(name, voice_emb)
            voice.speak("Voice captured successfully.")
            break
        else:
            voice.speak("I didn't hear anything clearly. Please speak louder.")
            time.sleep(1.0)
            
    brain.text_mem.ensure_user_exists(name)
    gui.set_emotion("HAPPY")
    voice.speak(f"You are all set, {name}!")


# --------------------------------------------------------------
# HANDLE VOICE TRAINING (Background)
# --------------------------------------------------------------
def handle_voice_training(voice_auth, brain, voice_assistant, user_name, existing_audio=None):
    print(f"[TRAINING] Voice enrollment started for {user_name}...")
    embedding = None
    
    if existing_audio:
        emb = voice_auth.get_embedding_from_data(existing_audio)
        if emb is not None:
            embedding = emb

    if embedding is None:
        pass 
    else:
        brain.bio_mem.save_voice(user_name, embedding)
        print(f"[TRAINING] Voice saved for {user_name} (Silent Mode).")
# --------------------------------------------------------------
# INTERACTION HANDLER (COMMAND PRIORITY + REGEX)
# --------------------------------------------------------------
def run_interaction(frame, detections, voice, brain, closest_depth, gui,
                    vision_system,
                    user_input=None, audio_data=None, mode="reactive",
                    voice_auth=None, my_id=0, 
                    pre_context=None, pre_ids=None):
    
    global current_active_user, interaction_id
    if my_id != interaction_id or voice.is_speaking:
        return

    # 1. SETUP DATA
    raw_identities = pre_ids if pre_ids else []
    rich_context = pre_context if pre_context else []
    known_users_present = [p['name'] for p in rich_context if p['name'] != "Unknown"]
    
    # 2. VOICE AUTH
    voice_match_name = None
    voice_score = 0.0
    if audio_data:
        emb = voice_auth.get_embedding_from_data(audio_data)
        match, score = voice_auth.compare_voices(brain.bio_mem.known_voice_embeddings,
                                                 emb, threshold=0.60)
        if match:
            voice_match_name = match
            voice_score = score

    # 3. DETERMINE SPEAKER
    primary_user = "Unknown"
    speaker_found = False

    for person in rich_context:
        if person['is_talking']:
            primary_user = person['name']
            speaker_found = True
            break
            
    if not speaker_found and voice_match_name and voice_match_name in known_users_present:
        primary_user = voice_match_name
        speaker_found = True

    if not speaker_found and voice_match_name and voice_score > 0.70:
        primary_user = voice_match_name
        speaker_found = True

    if not speaker_found:
        if len(rich_context) > 0:
            primary_user = rich_context[0]['name']
        else:
            primary_user = "Off-Screen Speaker"

    current_active_user = primary_user

    # 4. CONTEXT BUILDING
    context_str = f"People: {primary_user} (Active)"
    if primary_user == "Off-Screen Speaker":
        context_str += " [Audio Only]"
    for p in rich_context:
        if p['name'] != primary_user:
            context_str += f" | {p['name']} ({p['emotion']})"

    # 5. REINFORCEMENT
    if primary_user == voice_match_name and primary_user not in ["Unknown", "Off-Screen Speaker"]:
        for raw in raw_identities:
            if raw['name'] == primary_user:
                brain.bio_mem.reinforce_face(primary_user, raw['encoding'])
                break
        if audio_data:
            curr_emb = voice_auth.get_embedding_from_data(audio_data)
            if curr_emb is not None:
                brain.bio_mem.reinforce_voice(primary_user, curr_emb)

    # =========================================================================
    # 6. COMMANDS & ONBOARDING (BEFORE CHAT)
    # =========================================================================
    if user_input:
        clean_input = user_input.lower()
        trigger_wizard = False
        target_name = None

        # A. REGEX NAME EXTRACTION
        import re
        name_match = re.search(r"(?:i am|i'm|name is|this is) ([a-zA-Z]+)", clean_input)
        if name_match:
            potential_name = name_match.group(1).capitalize()
            if potential_name.lower() not in ["good", "sorry", "here", "ready",
                                              "friday", "happy", "fine"]:
                target_name = potential_name

        # B. BRAIN NAME EXTRACTION
        if not target_name:
            extracted = brain.extract_name(user_input)
            if extracted and extracted.lower() not in ["friday", "ai"]:
                target_name = extracted.strip()

        # CASE 1: Recalibrate
        if "recalibrate" in clean_input or "retrain" in clean_input or "reset voice" in clean_input:
            if target_name:
                voice.speak(f"Understood. Starting full factory reset for {target_name}.")
                trigger_wizard = True
            elif current_active_user not in ["Unknown", "Off-Screen Speaker"]:
                target_name = current_active_user
                voice.speak(f"Understood. Retraining profile for {target_name}.")
                trigger_wizard = True
            else:
                voice.speak("I need to know who you are first. Say 'Recalibrate [Name]'.")
                return

        # CASE 2: Register
        elif "register" in clean_input or "scan me" in clean_input:
            if target_name:
                trigger_wizard = True
            else:
                voice.speak("Who am I registering? Please say 'I am [Name]'.")
                return

        # CASE 3: "I am X"
        elif target_name:
            is_known = (target_name in brain.bio_mem.known_face_names)
            
            if not is_known:
                print(f"[SOCIAL] New User Detected: {target_name}")
                trigger_wizard = True
            
            elif current_active_user == "Unknown":
                print(f"[SOCIAL] Correction: User identified as {target_name}")
                voice.speak(f"Oh, I see you now, {target_name}.")
                current_active_user = target_name
                primary_user = target_name

        if trigger_wizard and target_name:
            onboarding_queue.append(target_name)
            return

    # 7. NORMAL CHAT RESPONSE
    if user_input:
        print(f"\n[USER] {user_input} (Context: {context_str})")
    gui.set_emotion("THINKING")
    
    response = brain.process(frame, user_input, detections,
                             context_str, None, closest_depth,
                             gui.emotion, mode)
    
    if my_id != interaction_id:
        return
    
    emo_match = re.search(r"\[EMOTION: (\w+)\]", response)
    if emo_match:
        emo = emo_match.group(1)
        if gui.emotion not in ["HUMMING", "BORED"]:
            gui.set_emotion(emo)
        response = response.replace(f"[EMOTION: {emo}]", "").strip()

    cleaned = response.strip()
    print(f"[FRIDAY] {cleaned}")
    voice.speak(cleaned)
# --------------------------------------------------------------
# AI AMBIENT DECISION MAKER (Smart Persona)
# --------------------------------------------------------------
def get_ai_ambient_choice(brain, last_user_text):
    """
    Decides ambient behavior based on context.
    Returns: "ACTION|CONTENT"
    """
    try:
        # If user told us to shut up recently, FORCE SILENCE
        if any(w in last_user_text.lower() for w in ["shut", "quiet", "silence", "stop"]):
            return "SILENCE|None"

        prompt = f"""
        You are 'Friday', an AI assistant. The user has been silent for a while.
        Decide what to do to keep the room alive.

        Last user interaction: "{last_user_text}"

        Choose ONE action from this list:
        1. BORED: Make a witty, slightly irritated comment.
        2. RANDOM: Say an interesting fact or philosophical question.
        3. MUSIC: Play a song (only if user wasn't angry).
        4. SYSTEM: Pretend to run a background diagnostic.
        5. SILENCE: Do nothing (if user seems busy).

        Return exactly: ACTION|Your Message
        """
        
        response = brain.model.generate_content(prompt).text.strip()
        
        if "|" not in response:
            return "SILENCE|None"
            
        return response
    except:
        return "SILENCE|None"


# --------------------------------------------------------------
# MAIN LOOP (UPDATED TO USE WEBCAM + GPU YOLO)
# --------------------------------------------------------------
def main():
    global interaction_id
    DM.clear_temp()
    gui = FridayUI()

    try:
        # CAMERA REPLACED — now using PC webcam
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # GPU YOLO detector
        detector = GPUDetector()

        voice = VoiceAssistant()
        brain = GeminiBrain()
        voice_auth = VoiceAuth()
        vision = VisionSystem()

    except Exception as e:
        print(f"[CRITICAL ERROR] Startup failed: {e}")
        sys.exit(1)

    print("\n[SYSTEM] FRIDAY 2.7 — GPU YOLO EDITION\n")

    frame_count = 0
    latest_detections = []
    latest_closest = None  # OAK depth removed
    face_center = (None, None)
    latest_vision_context = [] 
    latest_raw_ids = []

    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        # GPU YOLO DETECTION (REPLACES OAK)
        raw_dets, detections = detector.detect(frame)
        latest_detections = detections
        latest_closest = None  # no depth

        # FACE ID (existing logic)
        frame_count += 1
        if frame_count % 5 == 0:
            latest_raw_ids = brain.bio_mem.identify_all_faces(frame)

        # SMART VISION UPDATE (MediaPipe)
        latest_vision_context = vision.analyze(frame, latest_raw_ids)
        
        if latest_vision_context:
            face_center = latest_vision_context[0]["center"]
        else:
            face_center = (None, None)

        # GUI Update
        running = gui.update(face_center, voice.is_speaking)
        if not running:
            break

        # DRAW DEBUG HUD
        draw_debug_hud(frame, raw_dets, None, latest_vision_context)

        # LISTEN TRIGGER
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
            voice.stop()
            interaction_id += 1
            gui.set_emotion("LISTENING")
            gui.update(face_center, False)
            pygame.display.flip()
            
            user_text, user_audio = voice.listen()
            voice.last_interaction = time.time()

            if user_text:
                threading.Thread(
                    target=run_interaction,
                    args=(frame.copy(), latest_detections, voice, brain,
                          latest_closest, gui, vision,
                          user_text, user_audio,
                          "reactive", voice_auth, interaction_id,
                          latest_vision_context, latest_raw_ids),
                    daemon=True
                ).start()
            else:
                gui.set_emotion("IDLE")

        # AMBIENT MODE (AI DRIVEN)
        time_quiet = time.time() - voice.last_interaction
        is_music_playing = pygame.mixer.get_init() and pygame.mixer.music.get_busy()

        if time_quiet > config.AMBIENT_INTERVAL and not voice.is_speaking and not is_music_playing:
            last_text = "None"
            if brain.chat_history:
                for line in reversed(brain.chat_history):
                    if line.startswith("User:"):
                        last_text = line
                        break
            
            decision_str = get_ai_ambient_choice(brain, last_text)
            action, content = decision_str.split("|", 1)
            print(f"[AMBIENT AI] Action: {action} | Content: {content}")
            
            if action == "SILENCE":
                pass

            elif action == "BORED":
                gui.set_emotion("BORED")
                voice.speak(content)

            elif action == "RANDOM":
                gui.set_emotion("THINKING")
                voice.speak(content)

            elif action == "SYSTEM":
                gui.set_emotion("THINKING")
                voice.speak(content)

            elif action == "MUSIC":
                try:
                    songs = list(config.ASSETS_DIR.glob("*.mp3"))
                    if songs:
                        chosen = random.choice(songs)
                        if not pygame.mixer.get_init():
                            pygame.mixer.init(frequency=24000, buffer=2048)
                        pygame.mixer.music.load(str(chosen))
                        pygame.mixer.music.play()
                        
                        gui.set_emotion("HAPPY")
                        voice.speak(content)
                except:
                    pass

            voice.last_interaction = time.time()

        gui.clock.tick(60)

    pygame.quit()
    DM.clear_temp()

if __name__ == "__main__":
    main()
