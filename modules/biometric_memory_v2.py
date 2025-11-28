"""
=========================================================
  OPTIMIZED BIOMETRIC MEMORY (FACE + VOICE)
  
  Improvements over original:
  ✓ Multi-vector clustering per person
  ✓ Quality-aware storage (sharpness, lighting)
  ✓ Confidence-weighted reinforcement
  ✓ Temporal decay for old samples
  ✓ Multiple voice states (calm, excited, etc.)
=========================================================
"""

import json
import numpy as np
import face_recognition
import cv2
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


class BiometricCluster:
    """
    Stores multiple biometric samples for one person with quality metrics.
    """
    
    def __init__(self, person_name):
        self.name = person_name
        self.samples = []  # List of {vector, quality, timestamp, pose}
    
    def add_sample(self, vector, quality=0.8, pose="frontal"):
        """
        Add a new biometric sample with quality assessment.
        
        quality: 0.0-1.0 (based on sharpness, lighting, angle)
        pose: "frontal", "left", "right", "up", "down"
        """
        sample = {
            "vector": vector.tolist() if isinstance(vector, np.ndarray) else vector,
            "quality": quality,
            "timestamp": datetime.now().isoformat(),
            "pose": pose
        }
        self.samples.append(sample)
        
        # Keep max 15 samples per person (best quality + recent)
        if len(self.samples) > 15:
            self._prune_samples()
    
    def _prune_samples(self):
        """
        Keep the 15 best samples based on:
        - Quality score (40%)
        - Recency (30%)
        - Pose diversity (30%)
        """
        # Calculate age weights
        now = datetime.now()
        for s in self.samples:
            ts = datetime.fromisoformat(s["timestamp"])
            age_days = (now - ts).days
            age_weight = np.exp(-age_days / 30.0)  # Exponential decay
            
            # Combined score
            s["score"] = (s["quality"] * 0.4) + (age_weight * 0.3)
        
        # Ensure pose diversity (keep at least 2 of each pose if available)
        pose_counts = defaultdict(list)
        for s in self.samples:
            pose_counts[s["pose"]].append(s)
        
        kept = []
        for pose, samples in pose_counts.items():
            samples_sorted = sorted(samples, key=lambda x: x["score"], reverse=True)
            kept.extend(samples_sorted[:2])  # Keep top 2 per pose
        
        # Fill remaining slots with highest scores
        remaining = 15 - len(kept)
        if remaining > 0:
            other_samples = [s for s in self.samples if s not in kept]
            other_sorted = sorted(other_samples, key=lambda x: x["score"], reverse=True)
            kept.extend(other_sorted[:remaining])
        
        self.samples = kept[:15]
    
    def get_best_match_vector(self, target_vector, threshold=0.6):
        """
        Find the best matching sample from the cluster.
        Returns: (matched_sample, distance) or (None, 1.0)
        """
        if not self.samples:
            return None, 1.0
        
        best_dist = 1.0
        best_sample = None
        
        for sample in self.samples:
            vec = np.array(sample["vector"])
            dist = np.linalg.norm(vec - target_vector)
            
            # Weight by quality
            weighted_dist = dist * (2.0 - sample["quality"])
            
            if weighted_dist < best_dist:
                best_dist = weighted_dist
                best_sample = sample
        
        if best_dist < threshold:
            return best_sample, best_dist
        
        return None, best_dist
    
    def reinforce(self, new_vector, quality=0.8):
        """
        Confidence-weighted reinforcement learning.
        Updates the closest sample in the cluster.
        """
        if not self.samples:
            self.add_sample(new_vector, quality)
            return
        
        # Find closest sample
        best_idx = 0
        best_dist = 999
        
        for idx, sample in enumerate(self.samples):
            vec = np.array(sample["vector"])
            dist = np.linalg.norm(vec - new_vector)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        
        # Update with weighted average (higher quality = more influence)
        old_sample = self.samples[best_idx]
        old_vec = np.array(old_sample["vector"])
        old_quality = old_sample["quality"]
        
        # Weight formula: higher quality samples have more influence
        old_weight = old_quality * 0.7
        new_weight = quality * 0.3
        
        total_weight = old_weight + new_weight
        updated_vec = (old_vec * old_weight + new_vector * new_weight) / total_weight
        
        self.samples[best_idx]["vector"] = updated_vec.tolist()
        self.samples[best_idx]["quality"] = min(1.0, (old_quality + quality) / 2)
        self.samples[best_idx]["timestamp"] = datetime.now().isoformat()


class OptimizedBiometricMemory:
    """
    Manages face and voice biometric clusters for all known people.
    """
    
    def __init__(self, face_db_path="data/faces/face_clusters.json",
                 voice_db_path="data/voice/voice_clusters.json"):
        
        self.face_db_path = Path(face_db_path)
        self.voice_db_path = Path(voice_db_path)
        
        self.face_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.voice_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Storage: {person_name: BiometricCluster}
        self.face_clusters = {}
        self.voice_clusters = {}
        
        self.load_all()
    
    # ===================================================================
    # PERSISTENCE
    # ===================================================================
    def save_faces(self):
        data = {}
        for name, cluster in self.face_clusters.items():
            data[name] = {
                "samples": cluster.samples
            }
        
        with open(self.face_db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_voices(self):
        data = {}
        for name, cluster in self.voice_clusters.items():
            data[name] = {
                "samples": cluster.samples
            }
        
        with open(self.voice_db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_all(self):
        # Load faces
        if self.face_db_path.exists():
            try:
                with open(self.face_db_path, 'r') as f:
                    data = json.load(f)
                
                for name, cluster_data in data.items():
                    cluster = BiometricCluster(name)
                    cluster.samples = cluster_data["samples"]
                    self.face_clusters[name] = cluster
                
                print(f"[BIOMETRIC] Loaded {len(self.face_clusters)} face profiles")
            except Exception as e:
                print(f"[BIOMETRIC ERROR] Face load failed: {e}")
        
        # Load voices
        if self.voice_db_path.exists():
            try:
                with open(self.voice_db_path, 'r') as f:
                    data = json.load(f)
                
                for name, cluster_data in data.items():
                    cluster = BiometricCluster(name)
                    cluster.samples = cluster_data["samples"]
                    self.voice_clusters[name] = cluster
                
                print(f"[BIOMETRIC] Loaded {len(self.voice_clusters)} voice profiles")
            except Exception as e:
                print(f"[BIOMETRIC ERROR] Voice load failed: {e}")
    
    # ===================================================================
    # FACE OPERATIONS
    # ===================================================================
    def assess_face_quality(self, frame, face_location):
        """
        Calculate quality score for a face sample (0.0-1.0).
        
        Factors:
        - Sharpness (Laplacian variance)
        - Size (larger faces = better)
        - Lighting (histogram analysis)
        - Angle (frontal = best)
        """
        top, right, bottom, left = face_location
        face_img = frame[top:bottom, left:right]
        
        if face_img.size == 0:
            return 0.0
        
        # 1. Sharpness (Laplacian)
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500.0)  # Normalize
        
        # 2. Size score
        face_area = (right - left) * (bottom - top)
        size_score = min(1.0, face_area / 40000.0)  # Normalize to 200x200 optimal
        
        # 3. Lighting (histogram std dev)
        hist_std = np.std(gray)
        lighting_score = min(1.0, hist_std / 60.0)
        
        # Combined quality
        quality = (sharpness_score * 0.5) + (size_score * 0.3) + (lighting_score * 0.2)
        
        return max(0.1, min(1.0, quality))
    
    def detect_pose(self, face_landmarks):
        """
        Estimate head pose from facial landmarks.
        Returns: "frontal", "left", "right", "up", "down"
        """
        # Use nose tip vs face center to estimate angle
        # This is simplified; real implementation would use 3D pose estimation
        
        if not face_landmarks:
            return "frontal"
        
        # Placeholder logic (you'd need actual 3D pose estimation)
        # For now, just return frontal
        return "frontal"
    
    def save_face(self, name, encoding, frame=None, face_location=None):
        """
        Save a face encoding with quality assessment.
        """
        if name not in self.face_clusters:
            self.face_clusters[name] = BiometricCluster(name)
        
        quality = 0.8  # Default
        if frame is not None and face_location is not None:
            quality = self.assess_face_quality(frame, face_location)
        
        pose = "frontal"  # You can enhance this with actual pose detection
        
        self.face_clusters[name].add_sample(encoding, quality, pose)
        self.save_faces()
        
        print(f"[BIOMETRIC] Saved face for {name} (quality: {quality:.2f}, pose: {pose})")
    
    def reinforce_face(self, name, new_encoding, frame=None, face_location=None):
        """
        Reinforce existing face cluster with new observation.
        """
        if name not in self.face_clusters:
            self.save_face(name, new_encoding, frame, face_location)
            return
        
        quality = 0.8
        if frame is not None and face_location is not None:
            quality = self.assess_face_quality(frame, face_location)
        
        self.face_clusters[name].reinforce(new_encoding, quality)
        self.save_faces()
        
        print(f"[LEARNING] Reinforced face for {name} (quality: {quality:.2f})")
    
    def identify_face(self, encoding, threshold=0.6):
        """
        Identify a face using cluster-based matching.
        Returns: (name, confidence) or ("Unknown", 0.0)
        """
        best_name = "Unknown"
        best_confidence = 0.0
        
        for name, cluster in self.face_clusters.items():
            sample, dist = cluster.get_best_match_vector(encoding, threshold)
            
            if sample:
                # Convert distance to confidence (inverse)
                confidence = 1.0 - dist
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_name = name
        
        if best_confidence < 0.5:
            return "Unknown", 0.0
        
        return best_name, best_confidence
    
    def identify_all_faces(self, frame):
        """
        Scan frame for all faces and identify them.
        Returns: [{"name": "...", "encoding": ..., "center": (x, y), "confidence": 0.9}]
        """
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small)
        
        if not face_locations:
            return []
        
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        
        results = []
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Upscale coordinates
            center_x = int((left + right) / 2) * 4
            center_y = int((top + bottom) / 2) * 4
            
            # Identify
            name, confidence = self.identify_face(encoding)
            
            results.append({
                "name": name,
                "encoding": encoding,
                "center": (center_x, center_y),
                "confidence": confidence
            })
        
        return results
    
    # ===================================================================
    # VOICE OPERATIONS
    # ===================================================================
    def save_voice(self, name, embedding, quality=0.8, state="neutral"):
        """
        Save voice embedding with optional emotional state.
        state: "neutral", "excited", "tired", "angry", etc.
        """
        if name not in self.voice_clusters:
            self.voice_clusters[name] = BiometricCluster(name)
        
        self.voice_clusters[name].add_sample(embedding, quality, pose=state)
        self.save_voices()
        
        print(f"[BIOMETRIC] Saved voice for {name} (quality: {quality:.2f}, state: {state})")
    
    def reinforce_voice(self, name, new_embedding, quality=0.8):
        """
        Reinforce voice cluster.
        """
        if name not in self.voice_clusters:
            self.save_voice(name, new_embedding, quality)
            return
        
        self.voice_clusters[name].reinforce(new_embedding, quality)
        self.save_voices()
        
        print(f"[LEARNING] Reinforced voice for {name} (quality: {quality:.2f})")
    
    def identify_voice(self, embedding, threshold=0.60):
        """
        Identify speaker from voice embedding.
        Returns: (name, confidence) or (None, 0.0)
        """
        best_name = None
        best_similarity = 0.0
        
        for name, cluster in self.voice_clusters.items():
            sample, dist = cluster.get_best_match_vector(embedding, threshold=999)
            
            if sample:
                # For voice, we use cosine similarity (dot product of normalized vectors)
                vec = np.array(sample["vector"])
                similarity = np.dot(embedding, vec)
                
                # Weight by sample quality
                weighted_similarity = similarity * sample["quality"]
                
                if weighted_similarity > best_similarity:
                    best_similarity = weighted_similarity
                    best_name = name
        
        if best_similarity < threshold:
            return None, best_similarity
        
        return best_name, best_similarity
    
    # ===================================================================
    # UTILITIES
    # ===================================================================
    @property
    def known_face_names(self):
        """Compatibility with old code"""
        return list(self.face_clusters.keys())
    
    @property
    def known_voice_names(self):
        return list(self.voice_clusters.keys())
    
    def get_person_stats(self, name):
        """
        Get detailed statistics for a person's biometric data.
        """
        stats = {
            "name": name,
            "face_samples": 0,
            "voice_samples": 0,
            "avg_face_quality": 0.0,
            "avg_voice_quality": 0.0,
            "last_seen": None
        }
        
        if name in self.face_clusters:
            cluster = self.face_clusters[name]
            stats["face_samples"] = len(cluster.samples)
            if cluster.samples:
                stats["avg_face_quality"] = np.mean([s["quality"] for s in cluster.samples])
                stats["last_seen"] = max([s["timestamp"] for s in cluster.samples])
        
        if name in self.voice_clusters:
            cluster = self.voice_clusters[name]
            stats["voice_samples"] = len(cluster.samples)
            if cluster.samples:
                stats["avg_voice_quality"] = np.mean([s["quality"] for s in cluster.samples])
        
        return stats