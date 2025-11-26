import pygame
from . import config
import math
import time
import random
import numpy as np

def lerp(start, end, t):
    return start + (end - start) * t

def lerp_color(c1, c2, t):
    return (
        int(lerp(c1[0], c2[0], t)),
        int(lerp(c1[1], c2[1], t)),
        int(lerp(c1[2], c2[2], t))
    )

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.size = random.randint(2, 6)
        self.speed_y = random.uniform(0.5, 1.5)
        self.speed_x = random.uniform(-0.5, 0.5)
        self.color = color
        self.alpha = 255
        self.life = 1.0

    def update(self):
        self.y -= self.speed_y
        self.x += self.speed_x
        self.alpha -= 3
        self.life -= 0.01

    def draw(self, screen):
        if self.alpha > 0:
            s = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.color, self.alpha), (self.size, self.size), self.size)
            screen.blit(s, (self.x, self.y))

class FridayUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption("FRIDAY Interface")
        self.clock = pygame.time.Clock()
        
        # --- POSITIONS ---
        self.center_x = config.SCREEN_WIDTH // 2
        self.center_y = config.SCREEN_HEIGHT // 2
        
        # --- STATE & PHYSICS ---
        self.emotion = "IDLE"
        self.target_w = 140
        self.target_h = 150
        self.target_color = config.EYE_COLOR
        
        # Current Values (for smooth morphing)
        self.cur_w = 140
        self.cur_h = 150
        self.cur_col = list(config.EYE_COLOR)
        self.cur_x_off = 0
        self.cur_y_off = 0
        
        # Independent Eye Offsets (for rolling eyes)
        self.left_y_off = 0
        self.right_y_off = 0
        
        # Timers
        self.start_time = time.time()
        self.blink_timer = time.time() + 2
        self.saccade_timer = time.time() + 1
        self.is_blinking = False
        self.blink_progress = 0
        
        # Visual Flair
        self.particles = []
        self.font = pygame.font.SysFont("consolas", 18)

    def set_emotion(self, emotion_name):
        # Format: (Width, Height, Color)
        emotions = {
            "IDLE":      (140, 150, (0, 255, 255)),    # Cyan
            "LISTENING": (160, 160, (100, 255, 100)),  # Bright Green
            "THINKING":  (140, 140, (180, 100, 255)),  # Lavender
            "SPEAKING":  (150, 150, (0, 200, 255)),    # Blue
            "HAPPY":     (180, 120, (50, 255, 50)),    # Green (Squashed & Wide)
            "SURPRISED": (140, 220, (255, 255, 255)),  # White (Very Tall)
            "SKEPTICAL": (140, 60,  (255, 150, 0)),    # Orange (Narrow slit)
            "CONFUSED":  (140, 140, (255, 50, 255)),   # Purple
            "HUMMING":   (150, 40,  (50, 255, 180)),   # Chill Mint/Teal (Closed eyes)
            "BORED":     (140, 100, (100, 100, 150)),  # Grey-ish
        }
        
        self.emotion = emotion_name
        if emotion_name in emotions:
            t = emotions[emotion_name]
            self.target_w, self.target_h, self.target_color = t
        else:
            self.target_w, self.target_h, self.target_color = emotions["IDLE"]

    def draw_eye(self, x, y, w, h, color):
        # 1. Draw Glow
        s = pygame.Surface((int(w)+50, int(h)+50), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, 30), (0,0,w+50,h+50), border_radius=40)
        self.screen.blit(s, (x-25, y-25))
        
        # 2. Draw Core
        pygame.draw.rect(self.screen, color, (x, y, w, h), border_radius=15)
        
        # 3. Draw Shine (Only if eyes are open enough)
        if h > 30:
            shine_w, shine_h = w * 0.25, h * 0.25
            pygame.draw.rect(self.screen, (255,255,255), 
                           (x + w - shine_w - 10, y + 10, shine_w, shine_h), 
                           border_radius=5)

    def update(self, face_center, voice_active):
        self.screen.fill(config.BG_COLOR)
        t = time.time()
        dt = 0.1 # Animation speed
        
        # 1. Physics Interpolation
        self.cur_w = lerp(self.cur_w, self.target_w, dt)
        self.cur_h = lerp(self.cur_h, self.target_h, dt)
        self.cur_col = lerp_color(self.cur_col, self.target_color, dt/2)
        
        # 2. Base Positions
        left_x = self.center_x - 70 - self.cur_w 
        right_x = self.center_x + 70
        base_y = self.center_y - (self.cur_h // 2) + self.cur_y_off

        # 3. --- ANIMATION LOGIC ---
        
        # A. IDLE / BORED: Rolling Eyes
        if self.emotion == "BORED":
            roll = math.sin(t * 2) * 20
            self.left_y_off = roll
            self.right_y_off = -roll # Asymmetric
        else:
            self.left_y_off = lerp(self.left_y_off, 0, 0.1)
            self.right_y_off = lerp(self.right_y_off, 0, 0.1)

        # B. HUMMING: Swaying + Particles
        if self.emotion == "HUMMING":
            sway = math.sin(t * 4) * 10
            left_x += sway
            right_x += sway
            # Spawn particles
            if random.random() < 0.1:
                self.particles.append(Particle(
                    random.choice([left_x, right_x]) + self.cur_w/2, 
                    base_y, 
                    self.cur_col
                ))

        # C. HAPPY: Bouncing + Particles
        if self.emotion == "HAPPY":
            bounce = abs(math.sin(t * 10)) * 10
            base_y -= bounce
            if random.random() < 0.05:
                self.particles.append(Particle(
                    random.choice([left_x, right_x]) + self.cur_w/2, 
                    base_y, 
                    (255, 255, 100)
                ))

        # D. LISTENING: Pulse
        if self.emotion == "LISTENING":
            pulse = math.sin(t * 10) * 5
            self.cur_w += pulse
            self.cur_h += pulse

        # E. Saccades (Looking around)
        if t > self.saccade_timer:
            self.saccade_timer = t + random.uniform(0.5, 3.0)
            if self.emotion in ["IDLE", "THINKING"]:
                target_x_off = random.uniform(-10, 10)
                target_y_off = random.uniform(-10, 10)
                if face_center and face_center[0]:
                    target_x_off += np.interp(face_center[0], [0, config.CAM_WIDTH], [-50, 50])
                    target_y_off += np.interp(face_center[1], [0, config.CAM_HEIGHT], [-30, 30])
                self.cur_x_off = lerp(self.cur_x_off, target_x_off, 0.2)
                self.cur_y_off = lerp(self.cur_y_off, target_y_off, 0.2)

        # F. Blinking (Auto)
        if t > self.blink_timer and self.emotion not in ["HUMMING", "SKEPTICAL"]:
            self.is_blinking = True
            self.blink_progress = 0
            self.blink_timer = t + random.uniform(2, 6)
            
        blink_h_mod = 0
        if self.is_blinking:
            self.blink_progress += 0.5
            if self.blink_progress >= 3.14:
                self.is_blinking = False
            else:
                blink_h_mod = (self.cur_h * 0.9) * abs(math.sin(self.blink_progress))

        # 4. Render Particles
        for p in self.particles[:]:
            p.update()
            p.draw(self.screen)
            if p.life <= 0: self.particles.remove(p)

        # 5. Render Eyes
        # Apply all offsets
        final_ly = base_y + self.cur_y_off + self.left_y_off + (blink_h_mod/2)
        final_ry = base_y + self.cur_y_off + self.right_y_off + (blink_h_mod/2)
        final_lh = self.cur_h - blink_h_mod
        final_rh = self.cur_h - blink_h_mod
        
        self.draw_eye(left_x + self.cur_x_off, final_ly, self.cur_w, final_lh, self.cur_col)
        self.draw_eye(right_x + self.cur_x_off, final_ry, self.cur_w, final_rh, self.cur_col)
        
        # Status
        lbl = self.font.render(f"[{self.emotion}]", True, (80,80,80))
        self.screen.blit(lbl, (10, config.SCREEN_HEIGHT - 25))
        
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
        return True