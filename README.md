# Hand-Controlled 3D Model Viewer 🎮🖐️

A Python application that lets you **import and view 3D models with textures** and **control their rotation using hand gestures** via a webcam.  
Built with **OpenGL, PyWavefront, and MediaPipe**, featuring a simple **Tkinter GUI** for selecting models and textures.

---

## ✨ Features
- 🖼️ Load `.obj` 3D models with their `.mtl` and textures
- 🎨 Automatic texture binding support (`.jpg` / `.png`)
- 🖐️ Rotate the model using **hand tracking** (MediaPipe)
  - Pinch your **thumb + index finger** to activate rotation
  - Move your hand left/right → rotate Y-axis
  - Move your hand up/down → rotate X-axis
- 🖥️ Tkinter GUI for:
  - Selecting `.obj` models
  - Adding texture files
  - Launching the OpenGL 3D viewer
- 🎲 Fallback **3D cube** if no model is loaded

---

## 🛠️ Tech Stack
- [Python 3.10+](https://www.python.org/)
- [OpenCV](https://opencv.org/) (Webcam input + display)
- [MediaPipe](https://developers.google.com/mediapipe) (Hand tracking)
- [PyOpenGL](http://pyopengl.sourceforge.net/) (Rendering)
- [PyWavefront](https://github.com/pywavefront/PyWavefront) (OBJ/MTL loader)
- [Tkinter](https://wiki.python.org/moin/TkInter) (GUI)

---

## 📂 Project Structure
