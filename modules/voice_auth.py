from resemblyzer import VoiceEncoder
import numpy as np
import speech_recognition as sr
import warnings
import sys

warnings.filterwarnings("ignore", category=FutureWarning)



"""
=========================================================
   VOICE AUTHENTICATION MODULE (AUTONOMOUS VERSION)
=========================================================

This file provides:

  ✔ VoiceEncoder initialization
  ✔ Convert microphone or audio data → embeddings
  ✔ Normalize embedding vectors for stable similarity
  ✔ Voice comparison with adjustable threshold
  ✔ Used for both:
        - Voice identification (authentication)
        - Voice enrollment (saving new user embedding)

It is lightweight and designed for real-time operation.
"""


class VoiceAuth:
    def __init__(self):
        print("[SYSTEM] Loading Voice Biometrics...")

        try:
            # Load speaker encoder
            self.encoder = VoiceEncoder()
            self.r = sr.Recognizer()
        except Exception as e:
            print(f"[CRITICAL] VoiceAuth initialization failed: {e}")
            sys.exit(1)

    # --------------------------------------------------
    #  EMBEDDING FROM MIC (USED ONLY FOR TRAINING)
    # --------------------------------------------------
    def get_embedding_from_mic(self, duration=3.5):
        """
        Used when we need a clean voice sample for enrollment.
        """
        try:
            with sr.Microphone(sample_rate=16000) as source:
                self.r.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.r.record(source, duration=duration)
                return self._process_audio(audio)
        except Exception as e:
            print(f"[VOICE AUTH ERROR] Microphone capture failed: {e}")
            return None

    # --------------------------------------------------
    #  EMBEDDING FROM EXISTING AUDIO (LIVE CONVERSATION)
    # --------------------------------------------------
    def get_embedding_from_data(self, audio_data):
        """
        Extract embedding from STT audio captured during conversation.
        Returns None if audio is empty / invalid.
        """
        if audio_data is None:
            return None
        return self._process_audio(audio_data)

    # --------------------------------------------------
    #  INTERNAL AUDIO PROCESSOR
    # --------------------------------------------------
    def _process_audio(self, audio_data):
        """
        Convert raw audio → float32 PCM → speaker embedding.
        Normalizes vectors for reliable dot-product cosine similarity.
        """
        try:
            # Convert STT AudioData → raw PCM (16kHz, int16)
            raw = audio_data.get_raw_data(
                convert_rate=16000,
                convert_width=2
            )

            # Convert to numpy
            audio_np = np.frombuffer(raw, dtype=np.int16)

            # Convert to float32 [-1,1]
            audio_float = audio_np.astype(np.float32) / 32768.0

            # Generate speaker embedding
            emb = self.encoder.embed_utterance(audio_float)

            # Normalize for stable cosine similarity
            norm = np.linalg.norm(emb)
            if norm == 0:
                return None
            emb = emb / norm

            return emb

        except Exception:
            # Usually occurs if audio length < 0.5 sec
            return None

    # --------------------------------------------------
    #  VOICE MATCHING
    # --------------------------------------------------
    def compare_voices(self, known_embeddings, new_embedding, threshold=0.55):
        """
        Returns:
           (matched_name, similarity_score)

        If score < threshold → returns (None, score)

        We use dot-product similarity which behaves like cosine similarity
        because embeddings are normalized.
        """
        if not known_embeddings or new_embedding is None:
            return None, 0.0

        best_score = 0.0
        best_match = None

        for name, known_emb in known_embeddings.items():
            # Dot product of normalized vectors = cosine similarity
            similarity = np.dot(new_embedding, known_emb)

            if similarity > best_score:
                best_score = similarity
                best_match = name

        if best_score >= threshold:
            return best_match, best_score

        return None, best_score
