import os
import logging
import warnings
import sys
import speech_recognition as sr
import threading
import queue
import pygame
import torch
import time
import re
from melo.api import TTS
from . import config
from .memory import DM

# Silence logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

class VoiceAssistant:
    def __init__(self):
        self.r = sr.Recognizer()
        
        # Tuning for ghost inputs
        self.r.dynamic_energy_threshold = True
        # self.r.energy_threshold = 400
        self.r.pause_threshold = 1.0

        self.m = sr.Microphone()
        self.q = queue.Queue()
        self.is_speaking = False
        self.last_interaction = time.time()
        self.interrupt_flag = False
        
        self._ensure_mixer()
        
        print("[SYSTEM] Loading Voice Engine...")
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.tts = TTS(language='EN', device=dev)
            self.spk = self.tts.hps.data.spk2id['EN-US']
        except Exception as e:
            print(f"[CRITICAL] TTS Failed: {e}")
            sys.exit(1)

        threading.Thread(target=self._loop, daemon=True).start()
        
        with self.m as s: self.r.adjust_for_ambient_noise(s, duration=0.5)

    def _ensure_mixer(self):
        if not pygame.mixer.get_init():
            try: 
                pygame.mixer.init(frequency=24000, buffer=4096)
            except: pass

    def speak(self, text):
        if self.is_speaking: self.stop()
        
        # --- NEW PRONUNCIATION LOGIC ---
        # 1. Convert written acronym to phonetic name
        text = text.replace("E.D.I", "Edie").replace("EDI", "Edie")
        
        # 2. If "Friday" slips in, replace it with "Edie" (or remove it)
        text = text.replace("Friday", "Edie").replace("FRIDAY", "Edie")
        
        clean_text = re.sub(r'[^\w\s,!.?\']', '', text).strip()
        
        if clean_text: self.q.put(clean_text)

    def stop(self):
        self.interrupt_flag = True
        self._ensure_mixer()
        if pygame.mixer.get_init(): 
            pygame.mixer.music.stop()
        with self.q.mutex: self.q.queue.clear()
        self.is_speaking = False

    def _loop(self):
        while True:
            try:
                text = self.q.get(timeout=0.5)
            except queue.Empty:
                continue 
            
            self.is_speaking = True
            self.interrupt_flag = False
            fn = DM.get_temp_file()
            
            try:
                self.tts.tts_to_file(text, self.spk, fn, speed=1.1, quiet=True)
                
                if self.interrupt_flag: 
                    self.is_speaking = False
                    continue

                self._ensure_mixer()
                if os.path.exists(fn) and pygame.mixer.get_init():
                    pygame.mixer.music.load(fn)
                    pygame.mixer.music.play()
                    
                    while pygame.mixer.music.get_busy():
                        if self.interrupt_flag: 
                            pygame.mixer.music.stop()
                            break
                        time.sleep(0.05)
                        
                    pygame.mixer.music.unload()
            except Exception as e:
                print(f"[AUDIO ERROR] {e}")
                try: pygame.mixer.quit()
                except: pass
                self._ensure_mixer()
            finally:
                if os.path.exists(fn): 
                    try: os.remove(fn)
                    except: pass
                self.last_interaction = time.time()
                self.is_speaking = False
                try: self.q.task_done()
                except: pass

    def listen(self):
        with self.m as s:
            try:
                print("[DEBUG] Listening for audio...")
                # Reduce timeout slightly to make it snappier
                audio = self.r.listen(s, timeout=5, phrase_time_limit=10)
                
                print("[DEBUG] Audio captured. Sending to Google...")
                text = self.r.recognize_google(audio)
                
                print(f"[DEBUG] Google heard: '{text}'")
                return text, audio
            
            except sr.WaitTimeoutError:
                print("[DEBUG] Error: Listening Timed Out (No speech detected)")
                return None, None
            except sr.UnknownValueError:
                print("[DEBUG] Error: Google could not understand audio (Unclear speech)")
                return None, None
            except sr.RequestError as e:
                print(f"[DEBUG] Error: Could not request results from Google; {e}")
                return None, None
            except Exception as e: 
                print(f"[DEBUG] Critical Error in Listen: {e}")
                return None, None