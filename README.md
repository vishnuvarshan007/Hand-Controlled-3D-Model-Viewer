# 🖐️ Collaborative 3D Viewer with Gesture Control

An interactive **3D model viewer** that supports **hand gesture control**, **real-time collaboration over the network**, and **customizable environments**.  
Built with **Python, OpenGL, MediaPipe, and Socket.IO**, this project allows multiple users to **view, manipulate, and share 3D models** in sync.

---

## ✨ Features

- 🎮 **3D Model Viewer**
  - Load and render `.obj` models with materials and textures
  - Real-time OpenGL rendering with lighting & shading
  - Background customization (images or procedural sky)

- 🖐️ **Gesture Controls (MediaPipe Hands)**
  - Rotate, zoom, and pan models with finger pinches & hand gestures
  - Safety checks to prevent unstable manipulation
  - Smooth motion with configurable sensitivity

- 🌐 **Collaboration Server**
  - Built-in **Socket.IO server** for multi-user sessions
  - Broadcasts model updates and camera states across clients
  - Automatic asset packaging (OBJ, MTL, textures) into ZIP for sharing
  - Handles duplicate connections gracefully

- 📊 **HUD & System Monitoring**
  - Real-time FPS and frame time display
  - CPU, GPU, and memory usage overlay
  - Alerts for overloads and unsafe zoom/rotation

- 🎙️ **Voice Input/Output (optional)**
  - Speech recognition for command input (if `speech_recognition` installed)
  - Text-to-speech for notifications (via `pyttsx3`)

- 📝 **Sidebar Tools**
  - Notes and annotations
  - Drawing and additional UI features

---

## 📦 Dependencies

- **opencv-python** – Camera input  
- **mediapipe** – Hand tracking  
- **numpy, PIL** – Image processing  
- **PyOpenGL, pywavefront** – 3D rendering  
- **socketio, eventlet** – Networking  
- **psutil, GPUtil** – System stats  
- **tkinter** – File dialogs & UI  
- *(optional)* **speech_recognition, pyttsx3** – Voice input/output  

---

## 🚀 Run the application

```bash
python main.py


## Run the application:
python main.py

Controls:
Gestures

Thumb + Index → Rotate left
Thumb + Middle → Rotate right
Thumb + Ring → Tilt up
Thumb + Pinky → Tilt down
Pinch distance controls zoom

Sidebar
Open for notes & drawing tools

HUD
Monitor system usage, FPS, latency, and alerts

Collaborative Mode:
The host starts the server (auto-runs in background).
Other clients connect via the same IP/port (5000 by default).
Models and state updates sync automatically.

