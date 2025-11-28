"""
=========================================================
  MULTI-MODAL FUSION ENGINE
  
  Combines signals from:
  ✓ Vision (face, emotion, gestures, lip movement)
  ✓ Voice (speaker ID, tone, emotion)
  ✓ Context (spatial reasoning, temporal tracking)
  
  Provides unified person identification with confidence scoring.
=========================================================
"""

import numpy as np
from collections import deque
from datetime import datetime, timedelta


class PersonTracker:
    """
    Tracks a single person across time with multi-modal signals.
    """
    
    def __init__(self, person_id):
        self.id = person_id
        self.name = "Unknown"
        
        # Position tracking
        self.positions = deque(maxlen=10)  # Last 10 positions
        self.last_seen = datetime.now()
        
        # Identity confidence scores (multi-modal fusion)
        self.face_confidence = 0.0
        self.voice_confidence = 0.0
        self.spatial_confidence = 0.0
        
        # Behavioral signals
        self.is_speaking = False
        self.current_emotion = "neutral"
        self.detected_gestures = []
        
        # Temporal smoothing buffers
        self.emotion_history = deque(maxlen=5)
        self.speaking_history = deque(maxlen=3)
    
    def update_position(self, x, y):
        """Track person's movement through space."""
        self.positions.append((x, y, datetime.now()))
        self.last_seen = datetime.now()
        
        # Calculate spatial confidence (stable position = higher confidence)
        if len(self.positions) >= 3:
            recent = list(self.positions)[-3:]
            distances = []
            for i in range(len(recent) - 1):
                x1, y1, _ = recent[i]
                x2, y2, _ = recent[i + 1]
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distances.append(dist)
            
            avg_movement = np.mean(distances)
            # Low movement = high spatial confidence (person can't teleport)
            self.spatial_confidence = max(0.5, 1.0 - (avg_movement / 100.0))
    
    def update_identity(self, name, face_conf=0.0, voice_conf=0.0):
        """Update person's identity with confidence scores."""
        self.name = name
        self.face_confidence = face_conf
        self.voice_confidence = voice_conf
    
    def update_behavior(self, is_speaking, emotion, gestures):
        """Update behavioral signals with temporal smoothing."""
        self.speaking_history.append(is_speaking)
        self.emotion_history.append(emotion)
        
        # Smooth speaking detection (majority vote)
        self.is_speaking = sum(self.speaking_history) > len(self.speaking_history) / 2
        
        # Smooth emotion (most common in recent history)
        if self.emotion_history:
            from collections import Counter
            self.current_emotion = Counter(self.emotion_history).most_common(1)[0][0]
        
        self.detected_gestures = gestures
    
    def get_combined_confidence(self):
        """
        Calculate overall identity confidence using weighted fusion.
        
        Weights:
        - Face: 50%
        - Voice: 30%
        - Spatial: 20%
        """
        combined = (
            self.face_confidence * 0.5 +
            self.voice_confidence * 0.3 +
            self.spatial_confidence * 0.2
        )
        return combined
    
    def is_stale(self, max_age_seconds=5.0):
        """Check if tracker is outdated (person left frame)."""
        age = (datetime.now() - self.last_seen).total_seconds()
        return age > max_age_seconds


class MultiModalFusionEngine:
    """
    Main fusion engine that tracks all people and combines all signals.
    """
    
    def __init__(self):
        self.active_trackers = {}  # {tracker_id: PersonTracker}
        self.next_tracker_id = 0
        
        # Speaker history (who spoke when)
        self.speaker_history = deque(maxlen=20)
    
    def _find_nearest_tracker(self, x, y, max_distance=150):
        """
        Find the nearest active tracker to a position.
        Used for associating detections across frames.
        """
        nearest = None
        min_dist = max_distance
        
        for tid, tracker in self.active_trackers.items():
            if tracker.positions:
                last_x, last_y, _ = tracker.positions[-1]
                dist = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    nearest = tid
        
        return nearest
    
    def update(self, face_detections, vision_context, voice_match=None, voice_confidence=0.0):
        """
        Main fusion update.
        
        Args:
            face_detections: [{"name": "...", "center": (x, y), "confidence": 0.9, "encoding": ...}]
            vision_context: [{"name": "...", "center": (x, y), "is_talking": True, "emotion": "happy", "gestures": [...]}]
            voice_match: "PersonName" or None
            voice_confidence: 0.0-1.0
        
        Returns:
            List of identified persons with fused confidence scores.
        """
        # 1. Remove stale trackers
        stale_ids = [tid for tid, t in self.active_trackers.items() if t.is_stale()]
        for tid in stale_ids:
            del self.active_trackers[tid]
        
        # 2. Associate face detections with trackers (or create new ones)
        tracker_assignments = {}
        
        for face_det in face_detections:
            x, y = face_det["center"]
            name = face_det["name"]
            face_conf = face_det.get("confidence", 0.0)
            
            # Find nearest existing tracker
            tracker_id = self._find_nearest_tracker(x, y)
            
            if tracker_id is None:
                # Create new tracker
                tracker_id = self.next_tracker_id
                self.next_tracker_id += 1
                self.active_trackers[tracker_id] = PersonTracker(tracker_id)
            
            tracker = self.active_trackers[tracker_id]
            tracker.update_position(x, y)
            
            # Update identity (voice match logic)
            if voice_match and name == voice_match:
                # Voice confirms face ID
                tracker.update_identity(name, face_conf, voice_confidence)
            elif voice_match and name == "Unknown":
                # Voice identifies unknown face
                tracker.update_identity(voice_match, 0.0, voice_confidence)
            else:
                # Only face ID available
                tracker.update_identity(name, face_conf, 0.0)
            
            tracker_assignments[tracker_id] = face_det
        
        # 3. Merge with vision context (emotions, speaking, gestures)
        for vision_data in vision_context:
            vx, vy = vision_data["center"]
            
            # Find matching tracker
            tracker_id = self._find_nearest_tracker(vx, vy, max_distance=100)
            
            if tracker_id:
                tracker = self.active_trackers[tracker_id]
                tracker.update_behavior(
                    vision_data["is_talking"],
                    vision_data["emotion"],
                    vision_data.get("gestures", [])
                )
        
        # 4. Speaker identification (who's talking now?)
        current_speaker = None
        speaker_confidence = 0.0
        
        for tid, tracker in self.active_trackers.items():
            if tracker.is_speaking:
                conf = tracker.get_combined_confidence()
                if conf > speaker_confidence:
                    speaker_confidence = conf
                    current_speaker = tracker.name
        
        # If voice match but no visual speaker detected
        if voice_match and not current_speaker:
            current_speaker = voice_match
            speaker_confidence = voice_confidence
        
        # Log speaker history
        if current_speaker and current_speaker != "Unknown":
            self.speaker_history.append({
                "name": current_speaker,
                "timestamp": datetime.now(),
                "confidence": speaker_confidence
            })
        
        # 5. Build result
        results = []
        for tid, tracker in self.active_trackers.items():
            results.append({
                "name": tracker.name,
                "center": tracker.positions[-1][:2] if tracker.positions else (0, 0),
                "confidence": tracker.get_combined_confidence(),
                "is_speaking": tracker.is_speaking,
                "emotion": tracker.current_emotion,
                "gestures": tracker.detected_gestures,
                "face_confidence": tracker.face_confidence,
                "voice_confidence": tracker.voice_confidence,
                "spatial_confidence": tracker.spatial_confidence
            })
        
        # Add speaker metadata
        fusion_result = {
            "people": results,
            "current_speaker": current_speaker,
            "speaker_confidence": speaker_confidence
        }
        
        return fusion_result
    
    def get_conversation_participants(self, last_n_seconds=30):
        """
        Get list of people who spoke in the last N seconds.
        """
        cutoff = datetime.now() - timedelta(seconds=last_n_seconds)
        
        participants = set()
        for entry in self.speaker_history:
            if entry["timestamp"] > cutoff:
                participants.add(entry["name"])
        
        return list(participants)
    
    def get_primary_speaker(self):
        """
        Determine who's been speaking the most recently.
        """
        if not self.speaker_history:
            return None
        
        # Count recent speakers
        from collections import Counter
        recent = [e["name"] for e in list(self.speaker_history)[-5:]]
        
        if not recent:
            return None
        
        most_common = Counter(recent).most_common(1)[0][0]
        return most_common


class ContextualReasoningEngine:
    """
    Adds higher-level reasoning on top of fusion data.
    """
    
    def __init__(self):
        self.conversation_state = {
            "active_topic": None,
            "participants": [],
            "last_interaction": None
        }
    
    def infer_primary_user(self, fusion_result, voice_match=None):
        """
        Intelligently determine who the AI should respond to.
        """
        people = fusion_result["people"]
        
        # 1. If we have a voice match, but that person is NOT in the visual list
        #    it means they are speaking from off-screen.
        if voice_match:
            visible_names = [p["name"] for p in people]
            if voice_match not in visible_names:
                # If voice is strong enough, trust it
                if fusion_result["speaker_confidence"] > 0.5:
                     return voice_match, fusion_result["speaker_confidence"]

        if not people:
            # No face, but maybe voice?
            if voice_match: 
                return voice_match, fusion_result["speaker_confidence"]
            return "Unknown", 0.0
        
        # 2. Priority: Visual + Vocal (The person is right there and speaking)
        for person in people:
            if person["is_speaking"] and person["voice_confidence"] > 0.5:
                return person["name"], person["confidence"]
        
        # 3. Priority: Strong Voice Match (Even if lips aren't moving perfectly)
        if voice_match and fusion_result["speaker_confidence"] > 0.6:
            return voice_match, fusion_result["speaker_confidence"]
        
        # 4. Priority: Visually speaking (Lip movement only)
        for person in people:
            if person["is_speaking"] and person["face_confidence"] > 0.6:
                return person["name"], person["confidence"]
        
        # 5. Fallback: Person closest to center / highest confidence
        best_person = max(people, key=lambda p: p["confidence"])
        return best_person["name"], best_person["confidence"]
    
    def detect_conversation_shift(self, current_speaker, previous_speaker):
        """
        Detect when conversation switches between people.
        """
        if current_speaker != previous_speaker and current_speaker != "Unknown":
            return True
        return False
    
    def should_interrupt(self, fusion_result):
        """
        Determine if AI should stay silent (multiple people talking, etc.)
        """
        speaking_count = sum(1 for p in fusion_result["people"] if p["is_speaking"])
        
        # Don't interrupt if 2+ people talking
        if speaking_count >= 2:
            return False
        
        return True