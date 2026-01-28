# 🤖 E.D.I. — Enhanced Digital Intelligence  
### Social Vision Edition v2.0

E.D.I. (pronounced **Edie**) is a local-first, multi-modal social AI assistant that combines **computer vision**, **speaker recognition**, and a **persistent relational memory system**.

This project is not just a runtime application — it **builds a personal AI database over time**, storing biometric encodings and long-term knowledge locally on disk.

---

## ✨ Key Features

- Real-time object detection using **YOLOv8 (GPU / CUDA)**
- Face recognition without storing raw images
- Voice authentication using speaker embeddings
- Active speaker detection (vision + audio)
- Persistent memory using a Neo4j-style knowledge graph
- Fully local biometric storage (no cloud uploads)

---

## 🧠 Data Persistence Architecture (IMPORTANT)

E.D.I. continuously builds a structured “brain” inside the `data/` directory.  
Understanding this layout is essential for backups, migrations, and privacy.

### 📂 Directory Structure

```text
data/
├── faces/
│   └── face_clusters.json
│
├── voice/
│   └── voice_clusters.json
│
├── memory/
│   └── knowledge_graph.json
│
└── temp/


🧍 data/faces/face_clusters.json

Biometric face database

Stores numerical face encodings, not images

Each person has a cluster of vectors captured from:

Front

Left

Right

Up

Down

Includes quality/confidence scores to prevent learning blurry frames

Used for long-term identity recognition

🎤 data/voice/voice_clusters.json

Speaker recognition database

Stores 256-dimension voice embeddings

Generated during vocal authorization

Allows identity recognition even when the user is off-camera

Used by the VoiceAuth module

🧠 data/memory/knowledge_graph.json

Persistent relational memory (Edie’s “soul”)

Neo4j-style graph stored as JSON

Stores:

Facts: Parth → LIKES → Coffee

Relationships: User → WORKS_ON → Project

Episodic summaries of past interactions

Enables long-term personalization across sessions

⚠️ Deleting this file resets Edie’s personality and memory.

🧪 data/temp/

Temporary runtime storage

Used for:

Generated .wav files (TTS)

Short-lived intermediate data

Automatically cleared by the system

Safe to delete at any time

🛠️ System Requirements
Hardware

NVIDIA GPU (recommended for YOLOv8)

CUDA-compatible drivers

Webcam + microphone

Software

Python 3.10

Conda (recommended)

NVIDIA drivers + CUDA toolkit

📦 requirements.txt (Version-Pinned)

Due to compatibility issues between MediaPipe and NumPy 2.x, strict version pinning is required.

Ensure your requirements.txt contains at least the following:

numpy==1.26.4
mediapipe==0.10.11
opencv-python==4.8.0.74
protobuf==3.20.3
ultralytics
google-generativeai
resemblyzer
face-recognition

🚀 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/EDI.git
cd EDI

2️⃣ Create Conda Environment
conda create -n ai_lab python=3.10
conda activate ai_lab

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Hardware Check

Ensure NVIDIA drivers are installed

Verify CUDA is available:

nvidia-smi

5️⃣ Run E.D.I.
python main.py

🎮 First-Run Onboarding

On detecting a new person, E.D.I. initiates identity calibration:

Face Scan

Captures 5 head orientations

Builds a face encoding cluster

Voice Authorization

Records a short authorization phrase

Generates a 256-D speaker embedding

Knowledge Graph Node Creation

Identity is added to persistent memory

🔒 Privacy & Security Notice

This project stores biometric data locally.

IMPORTANT:

If you plan to publish your fork or repository:

DO NOT commit the data/ directory

DO NOT commit .env files

Your face, voice, and memory data should remain private.

🛡️ Recommended .gitignore
# Virtual environments
.env
.venv
ai_lab/

# Biometric & memory data
data/

# Python cache
__pycache__/
*.pyc

# OS files
.DS_Store
Thumbs.db

🧩 Project Philosophy

E.D.I. is designed as a stateful, embodied AI system, not a stateless chatbot.
Its intelligence emerges over time through perception, memory, and interaction.
