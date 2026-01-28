🤖 E.D.I. — Enhanced Digital Intelligence
Social Vision Edition v2.0

E.D.I. (phonetically "Edie") is a high-performance, multi-modal social AI assistant. Built to bridge the gap between computer vision and conversational intelligence, Edie doesn't just "chat"—she observes her environment, recognizes individuals by their biometrics, and remembers personal history through a relational memory system.

⚡ Core Engine Capabilities
👁️ Social Vision
GPU YOLOv8 Integration: Real-time object detection forced to CUDA for maximum FPS.

Active Speaker Detection: Combines MediaPipe lip-tracking with VoiceAuth to identify exactly who is talking.

Emotion & Gesture Mapping: Analyzes facial expressions and hand gestures (👍, Waving) to adapt her personality.

🧠 Advanced Memory (Persistence Layer)
Edie's brain is stored in ./data/, organized into three distinct layers:

Biometric Clusters: Stores high-fidelity face encodings and voice embeddings in person-specific data clusters.

Knowledge Graph: A Neo4j-style relational memory that maps entities and relationships (e.g., Parth → LIKES → Coffee).

Episodic Logging: Summarizes past interactions to maintain long-term context across sessions.

🛠️ Environment "Just-Works" Setup
Due to the complex nature of the ai_lab environment (balancing MeloTTS and MediaPipe), the following specific versions are required to prevent the NumPy 2.x crash:

Bash
# 1. Clean the environment
pip uninstall mediapipe opencv-python numpy -y

# 2. Install "The Golden Stack"
pip install "numpy<2" "protobuf<4"
pip install mediapipe==0.10.11 opencv-python==4.8.0.74 
pip install ultralytics face-recognition deepface google-generativeai resemblyzer
🎮 Initial Calibration (Onboarding)
When E.D.I. detects a new presence, the Onboarding Protocol initiates a biometric "handshake":

Spatial Scan: Captures 5 head angles (Front, Left, Right, Up, Down) to build your face cluster.

Vocal Sync: Generates a 256-dimension speaker embedding from your authorization phrase.

Identity Birth: Creates your unique node in the Knowledge Graph for persistent recognition.

📂 Data Topology
YAML
data/

  ├── faces/
  
  │   └── face_clusters.json    # Mathematical face maps (Not raw photos)
  
  ├── voice/
  
  │   └── voice_clusters.json   # Speaker embeddings (Vocal fingerprints)
  
  ├── memory/
  
  │   └── knowledge_graph.json  # Edie's "Soul" (Facts & Relationships)
  
  └── temp/                     # Managed scratchpad for TTS generation
  
🔒 Privacy & Security
No Cloud Biometrics: All face and voice data are stored locally on your machine.

Safe GitHubbing: Ensure your data/ folder and .env (API Keys) are listed in your .gitignore.
