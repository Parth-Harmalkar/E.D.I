# 🤖 E.D.I. — Enhanced Digital Intelligence
## Social Vision Edition v2.0

E.D.I. (pronounced **Edie**) is a local-first, multi-modal social AI assistant that combines computer vision, speaker recognition, and a persistent relational memory system.

This project is not just a runtime application — it builds a personal AI database over time, storing biometric encodings and long-term knowledge locally on disk.

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
├── voice/
│   └── voice_clusters.json
├── memory/
│   └── knowledge_graph.json
└── temp/
```


## 🧍 data/faces/face_clusters.json
Biometric Face Database

Stores numerical face encodings (no raw images)

Each person has a cluster of vectors captured from:

Front

Left

Right

Up

Down

Tracks quality and confidence scores to prevent learning blurry frames

Used for long-term identity recognition

## 🎤 data/voice/voice_clusters.json
Speaker Recognition Database

Stores 256-dimensional voice embeddings

Generated during vocal authorization

Allows identity recognition even when the user is off-camera

Used by the VoiceAuth module

## 🧠 data/memory/knowledge_graph.json
Persistent Relational Memory (Edie’s “Soul”)

Neo4j-style relational graph stored as JSON

Stores:

Facts (e.g., `Parth → LIKES → Coffee`)

Relationships (e.g., `User → WORKS_ON → Project`)

Episodic summaries of past interactions

Enables long-term personalization across sessions

⚠️ Deleting this file resets Edie’s personality and memory.

## 🧪 data/temp/
Temporary Runtime Storage

Used for:

Generated `.wav` files (Text-to-Speech)

Short-lived intermediate runtime data

Automatically cleared by the system

Safe to delete at any time

🛠️ System Requirements
Hardware
NVIDIA GPU (recommended for YOLOv8)

CUDA-compatible drivers

Webcam and microphone

Software
Python 3.10

Conda (recommended)

NVIDIA drivers + CUDA toolkit

📦 requirements.txt (Version-Pinned)
Due to compatibility issues between MediaPipe and NumPy 2.x, strict version pinning is required.
```
numpy==1.26.4
mediapipe==0.10.11
opencv-python==4.8.0.74
protobuf==3.20.3
ultralytics
google-generativeai
resemblyzer
face-recognition
```
⚠️ Installing NumPy 2.x will crash MediaPipe.

## 🚀 Installation & Setup
1️⃣ Clone the Repository
```
git clone https://github.com/yourusername/EDI.git
cd EDI
```
2️⃣ Create Conda Environment
```
conda create -n ai_lab python=3.10
conda activate ai_lab
```
3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
4️⃣ Hardware Check
Ensure NVIDIA drivers are installed and CUDA is available:
```
nvidia-smi
```
5️⃣ Run E.D.I.
```
python main.py
```
🎮 First-Run Onboarding
When E.D.I. detects a new person, it initiates identity calibration.

## Face Scan
Captures five head orientations

Builds a face encoding cluster

## Voice Authorization
Records a short authorization phrase

Generates a 256-dimensional speaker embedding

## Knowledge Graph Initialization
Creates a persistent identity node

Links future memories and preferences

## 🔒 Privacy & Security Notice
This project stores biometric data locally on disk.

## IMPORTANT
If you plan to publish your fork or make the repository public:

DO NOT commit the `data/` directory

DO NOT commit `.env` files

Your face data, voice data, and memory graph should remain private.

## 🛡️ Recommended .gitignore
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

