import os
import sys
import time
import math
import threading
import atexit
from dataclasses import dataclass
import random
import cv2
import numpy as np
from PIL import Image
import socketio
import eventlet
import psutil
import io
import GPUtil
import tempfile, zipfile, base64


# MediaPipe
import mediapipe as mp
mp_hands = mp.solutions.hands

# OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# OBJ loader + optional faster draw
import pywavefront
from pywavefront.visualization import draw as pw_draw

# Tkinter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

# Try to import speech_recognition (optional)
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

# Try to import pyttsx3 for TTS (optional)
try:
    import pyttsx3
    TTS_ENGINE = pyttsx3.init()
    def speak(text: str):
        try:
            TTS_ENGINE.say(text)
            TTS_ENGINE.runAndWait()
        except Exception:
            pass
except Exception:
    def speak(text: str):
        # fallback: do nothing if pyttsx3 not available
        pass

# ---------------- configuration ----------------
TARGET_FPS = 60
CAM_DOWNSCALE = (640, 480)
MEDIAPIPE_EVERY_N = 2
SHOW_DEBUG_WINDOW = False
PINCH_PIXEL_THRESHOLD = 40
current_model_zip = None

# ---------------- HUD state ----------------
last_alert_time = 0
last_frame_time = time.time()
fps = 0
frame_time_ms = 0.0
fps_display, frame_ms = 0.0, 0.0
connected_clients = {}

sidebar_visible = False
sidebar_alpha = 0.0
sidebar_target_alpha = 0.0
sidebar_last_time = time.time()

SYSTEM_STATS = {
    "cpu": 0.0,
    "mem": 0.0,
    "gpu": -1.0
}

# ---------------- Collaboration Server ----------------
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000
sio = socketio.Server(cors_allowed_origins="*")
app = socketio.WSGIApp(sio)


# ---------------- view state ----------------
@dataclass
class ViewState:
    rot_x: float = 0.0   # Pitch (up/down tilt)
    rot_y: float = 0.0   # Yaw (left/right rotation)
    pan_x: float = 0.0
    pan_y: float = 0.0
    dist: float = 8.0

@dataclass
class Smoothing:
    alpha: float = 0.25
    idx_x: float = 0.0
    idx_y: float = 0.0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

last_broadcast_time = 0
def broadcast_state(throttle=0.05):
    global last_broadcast_time
    now = time.time()
    if now - last_broadcast_time < throttle:
        return
    try:
        sio.emit("state", shared_state)
        last_broadcast_time = now
    except Exception as e:
        print("[broadcast_state] emit error:",e)


# ---------------- Collaboration Server ----------------


def package_model_with_assets(obj_path):
    print("[server] Packaging model and assets...")
    """Create a zip containing OBJ, MTL, and textures."""
    base_dir = os.path.dirname(obj_path)
    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # OBJ
        zipf.write(obj_path, os.path.basename(obj_path))
        mtl_file = None
        with open(obj_path, "r") as f:
            for line in f:
                if line.lower().startswith("mtllib"):
                    mtl_file = line.split()[1].strip()
                    break
        if mtl_file and os.path.exists(os.path.join(base_dir, mtl_file)):
            mtl_path = os.path.join(base_dir, mtl_file)
            zipf.write(mtl_path, os.path.basename(mtl_path))
            # Textures from MTL
            with open(mtl_path, "r") as mf:
                for line in mf:
                    if line.lower().startswith("map_kd"):
                        tex = line.split()[1].strip()
                        tex_path = os.path.join(base_dir, tex)
                        if os.path.exists(tex_path):
                            zipf.write(tex_path, os.path.basename(tex_path))
    zip_bytes.seek(0)
    return zip_bytes.read()

current_model_version = 0

def send_model_to_clients(obj_path):
    print("[server] Sending model to clients:", obj_path)
    global current_model_zip, current_model_version
    try:
        current_model_zip = package_model_with_assets(obj_path)
        current_model_version += 1
        encoded = base64.b64encode(current_model_zip).decode("utf-8")
        sio.emit("model_data", {"zip": encoded, "version": current_model_version})
        print(f"[server] Broadcasted new model v{current_model_version}: {obj_path}")
    except Exception as e:
        print("[server] Failed to package model:",e)

@sio.event
def connect(sid, environ, auth=None):
    ip_port = f"{environ.get('REMOTE_ADDR', 'unknown')}:{environ.get('REMOTE_PORT', 'unknown')}"
    print(f"[server] Client connected: sid={sid}, ip_port={ip_port}, headers={environ.get('HTTP_USER_AGENT', 'unknown')}")
    # Check if client IP/port already received the current model version
    for existing_sid, info in list(connected_clients.items()):
        if info["ip_port"] == ip_port and info["version"] >= current_model_version:
            print(f"[server] Client {ip_port} already has model version {info['version']}, skipping model_data for sid={sid}")
            connected_clients[sid] = {"version": info["version"], "ip_port": ip_port}
            sio.emit("state", shared_state, to=sid)
            return
    connected_clients[sid] = {"version": 0, "ip_port": ip_port}
    if current_model_zip and connected_clients[sid]["version"] < current_model_version:
        encoded = base64.b64encode(current_model_zip).decode("utf-8")
        print(f"[server] Sending model to client {sid}, version: {current_model_version}")
        sio.emit("model_data", {"zip": encoded, "version": current_model_version}, to=sid)
        connected_clients[sid]["version"] = current_model_version
    else:
        print(f"[server] No model loaded or client already has model (version {connected_clients[sid]['version']}), skipping model_data for sid={sid}")
    sio.emit("state", shared_state, to=sid)
# Client-side handler (runs in all instances, including host)

@sio.on("state")
def on_state(sid, data):
    global shared_state
    # Validate and sanitize incoming data here if you want
    shared_state.update(data)
    shared_state["server_time"] = time.time()
    # Broadcast new state to all other clients except sender
    sio.emit("state", shared_state, skip_sid=sid)

@sio.on("state")
def on_state(sid, data):
    global shared_state
    shared_state.update(data)
    shared_state["server_time"] = time.time()
    # Broadcast to all other clients except sender
    sio.emit("state", shared_state, skip_sid=sid)


@sio.on("state")
def handle_state(sid, data):
    global shared_state
    # Run safety checks
    alerts = run_safety_checks(shared_state, data)
    data["alerts"] = alerts

    # Update server copy of state
    shared_state.update(data)
    shared_state["server_time"] = time.time()

    print(f"[server] State update from {sid}: {data}")

    # Broadcast new state to all clients (except sender)
    sio.emit("state", shared_state, skip_sid=sid)

@sio.on("model_data")
def on_model_data(data):
    print("[client] Received model data from server")
    try:
        raw = base64.b64decode(data["zip"])
        tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        tmp_zip.write(raw)
        tmp_zip.close()
        extract_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(tmp_zip.name, "r") as zf:
            zf.extractall(extract_dir)
        # Find obj inside extracted
        for f in os.listdir(extract_dir):
            if f.endswith(".obj"):
                obj_path = os.path.join(extract_dir, f)
                load_obj(obj_path)  # reuse your loader
                break
        print("[client] Model updated from server")
    except Exception as e:
        print("[client] Failed to load model from server:", e)

def start_server():
    print(f"[server] Running on {SERVER_HOST}:{SERVER_PORT}")
    eventlet.wsgi.server(eventlet.listen((SERVER_HOST, SERVER_PORT)), app)

threading.Thread(target=start_server, daemon=True).start()



def parse_obj(path):
    print(f"Parsing OBJ: {path}")
    vertices, faces = [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):  # vertex
                _, x, y, z = line.split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):  # face
                parts = [p.split("/")[0] for p in line.split()[1:]]
                faces.append([int(p) - 1 for p in parts])  # 0-indexed
    return {"vertices": vertices, "faces": faces}

def choose_model():
    print("Choose model file...")
    path = filedialog.askopenfilename(filetypes=[("OBJ files", "*.obj")])
    if path:
        load_obj(path)             # load locally
        send_model_to_clients(path)  # broadcast


# ---------------- global state ----------------
class AppState:
    def __init__(self):
        self.model_scene = None
        self.mesh_ready = False
        self.model_name = "(none)"
        self.model_center = np.array([0.0,0.0,0.0], dtype=np.float32)
        self.model_scale = 1.0

        self.view = ViewState()
        self.smooth = Smoothing(alpha=0.25)

        self.cap = None
        self.hands = None

        self.rotate_sensitivity = 2.5
        self.zoom_sensitivity = 0.5

        self.bg_mode = 0
        self.wireframe = False

        self._last_processed = None
        self._last_frame = None
        self._lock = threading.Lock()

        self.running = True
        self.current_fps = 0.0
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.current_gesture = "None"

        self.bg_image_path = None
        self.bg_tex_id = None
        self.bg_img_size = (0, 0)

        # Sidebar state
        self.sidebar_open = False
        self.sidebar_anim_x = -300
        self.sidebar_target_x = -300
        self.sidebar_speed = 25

        # numeric selection (1,2,3) shown in sidebar; None when nothing selected
        self.sidebar_selection = None
        self.last_note_preview = ""
        self.is_recording_note = False
        self.last_action_message = ""

STATE = AppState()

NOTES_DIR = os.path.join(os.getcwd(), "notes")
os.makedirs(NOTES_DIR, exist_ok=True)

# Shared state that all clients will see
shared_state = {
    "rot_x": STATE.view.rot_x,
    "rot_y": STATE.view.rot_y,
    "dist": STATE.view.dist,
    "alerts": [],
}


def get_system_stats():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    gpu = 0
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0].load * 100
    except Exception as e:
        gpu = -1  # means unavailable
    return cpu, mem, gpu

def poll_system_stats():
    while True:
        try:
            SYSTEM_STATS["cpu"] = psutil.cpu_percent(interval=0.5)
            SYSTEM_STATS["mem"] = psutil.virtual_memory().percent
            gpu_load = -1.0
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_load = gpus[0].load * 100
            except Exception:
                gpu_load = -1.0
            SYSTEM_STATS["gpu"] = gpu_load
        except Exception:
            SYSTEM_STATS["cpu"] = 0.0
            SYSTEM_STATS["mem"] = 0.0
            SYSTEM_STATS["gpu"] = -1.0
        time.sleep(1.0)

threading.Thread(target=poll_system_stats, daemon=True).start()


def update_fps():
    """Update FPS and frame time every 0.5s."""
    global last_fps_time, frame_count, fps_display, frame_ms
    frame_count += 1
    now = time.time()
    if now - last_fps_time >= 0.5:
        fps_display = frame_count / (now - last_fps_time)
        frame_ms = 1000.0 / fps_display if fps_display > 0 else 0
        frame_count = 0
        last_fps_time = now
    return fps_display, frame_ms

# ---------------- model helpers ----------------
def compute_bbox(scene):
    mn = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    mx = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    for mesh in scene.mesh_list:
        for mat in mesh.materials:
            verts = getattr(mat, 'vertices', [])
            for i in range(0, len(verts), 8):
                x,y,z = verts[i], verts[i+1], verts[i+2]
                mn = np.minimum(mn, [x,y,z])
                mx = np.maximum(mx, [x,y,z])
    return mn, mx

def load_obj(path):
    try:
        scene = pywavefront.Wavefront(path, collect_faces=True, parse=True, strict=False, encoding='utf-8')
        STATE.model_scene = scene
        STATE.model_name = os.path.basename(path)
        mn,mx = compute_bbox(scene)
        center = (mn + mx) / 2.0
        size = np.linalg.norm(mx - mn)
        if size == 0: size = 1.0
        STATE.model_center = center
        STATE.model_scale = 4.0 / size
        STATE.mesh_ready = True
        msg = f"Model loaded: {STATE.model_name}"
        print(msg)
        speak(msg)
    except Exception as e:
        STATE.mesh_ready = False
        print("Failed to load OBJ:", e)
        messagebox.showerror("Load Error", f"Failed to load OBJ: {e}")

def load_background_image(path):
    """Loads image from 'path', uploads it as an OpenGL texture, sets STATE.bg_tex_id and bg_img_size."""
    try:
        img = Image.open(path).convert('RGB')
        img = img.transpose(Image.FLIP_TOP_BOTTOM) # OpenGL expects bottom-to-top
        img_data = np.array(img, dtype=np.uint8)
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        STATE.bg_tex_id = tex_id
        STATE.bg_img_size = (img.width, img.height)
        STATE.bg_image_path = path
        print(f"[Background] Loaded background image: {path}")
    except Exception as e:
        print("Failed to load background image:", e)
        messagebox.showerror("Background Error", f"Failed to load background image: {e}")

CLOUD_TEX_ID = None

def generate_cloud_texture(size=256):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            # base gradient sky (blue to white)
            t = y / size
            r = int(135 + 120*t)
            g = int(206 + 49*t)
            b = int(235 + 20*t)
            # add soft white cloud patches
            if random.random() < 0.005:
                r = g = b = 255
            img[y, x] = (r, g, b)
    return img

def init_sky_texture():
    global CLOUD_TEX_ID
    img = generate_cloud_texture()
    CLOUD_TEX_ID = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, CLOUD_TEX_ID)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0],
                 0, GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

def draw_sky_background():
    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, 1, 0, 1, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glEnable(GL_TEXTURE_2D)
    if STATE.bg_tex_id:   # --- use user's texture if set
        glBindTexture(GL_TEXTURE_2D, STATE.bg_tex_id)
    else:
        glBindTexture(GL_TEXTURE_2D, CLOUD_TEX_ID)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(0, 0)
    glTexCoord2f(1, 0); glVertex2f(1, 0)
    glTexCoord2f(1, 1); glVertex2f(1, 1)
    glTexCoord2f(0, 1); glVertex2f(0, 1)
    glEnd()

    glDisable(GL_TEXTURE_2D)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)



# ---------------- safety checks ----------------
def run_safety_checks(prev_state, new_state):
    alerts = []
    # 1. Zoom too close or too far
    if new_state["dist"] < 5.0:
        alerts.append("⚠️ Too close – clipping risk")
    if new_state["dist"] > 30.0:
        alerts.append("⚠️ Too far – model may disappear")


    # 2. Rotation too fast (threshold: >15 deg/frame)
    if abs(new_state["rot_x"] - prev_state["rot_x"]) > 15 or abs(new_state["rot_y"] - prev_state["rot_y"]) > 15:
        alerts.append("⚠️ Unstable manipulation (rotation too fast)")

    return alerts

# ---------------- OpenGL rendering ----------------
WINDOW_W, WINDOW_H = 640, 480

def init_gl(w, h):
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat*4)(5.0,6.0,8.0,1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  (GLfloat*4)(0.95,0.95,0.95,1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (GLfloat*4)(0.6,0.6,0.6,1.0))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glShadeModel(GL_SMOOTH)
    resize_gl(w,h)
    init_sky_texture()  # always have default

def resize_gl(w,h):
    if h==0: h=1
    glViewport(0,0,w,h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(w)/float(h), 0.05, 100.0)
    glMatrixMode(GL_MODELVIEW)

def draw_hud():
    global last_alert_time, last_frame_time, fps, frame_time_ms

    # --- FPS / frame timing ---
    now = time.time()
    dt = now - last_frame_time
    last_frame_time = now
    if dt > 0:
        fps = int(1.0 / dt)
        frame_time_ms = dt * 1000.0
    cpu = SYSTEM_STATS["cpu"]
    mem = SYSTEM_STATS["mem"]
    gpu = SYSTEM_STATS["gpu"]


    # Projection setup
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, WINDOW_W, 0, WINDOW_H)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_LIGHTING)

    def draw_text(x, y, text, color=(1, 1, 1), size=0.15):
        glPushAttrib(GL_CURRENT_BIT)  # isolate color state
        glColor3f(*color)
        glPushMatrix()
        glTranslatef(x, y, 0)
        glScalef(size, size, size)
        for ch in text:
            glutStrokeCharacter(GLUT_STROKE_ROMAN, ord(ch))
        glPopMatrix()
        glPopAttrib()

    # --- Status ---
    draw_text(20, WINDOW_H - 80, f"Gesture: {STATE.current_gesture}", color=(0.6, 1, 0.6))
    draw_text(20, WINDOW_H - 110, f"Action: {STATE.last_action_message}", color=(1, 1, 0.5))

    # --- Alerts ---
    if shared_state.get("alerts"):
        t = time.time()
        pulse = (abs((t * 2) % 2 - 1)) * 0.5 + 0.5  # 0.5–1.0 pulsing
        for i, msg in enumerate(shared_state["alerts"]):
            draw_text(
                20, 40 + i * 40,
                msg,
                color=(1, pulse * 0.3, pulse * 0.3),  # pulsing red
                size=0.18
            )

    # --- Developer Info ---
    draw_text(WINDOW_W - 220, WINDOW_H - 40, f"FPS: {fps}", color=(0.8, 0.8, 0.8), size=0.15)
    draw_text(WINDOW_W - 220, WINDOW_H - 70, f"Frame: {frame_time_ms:.2f} ms", color=(0.8, 0.8, 0.8), size=0.15)
    draw_text(WINDOW_W - 220, WINDOW_H - 100, f"CPU: {cpu:.1f}%", color=(0.8, 0.8, 0.8), size=0.15)
    draw_text(WINDOW_W - 220, WINDOW_H - 140, f"MEM: {mem:.1f}%", color=(0.8, 0.8, 0.8), size=0.15)
    draw_text(WINDOW_W - 220, WINDOW_H - 170, f"GPU: {gpu:.1f}%", color=(0.8, 0.8, 0.8), size=0.15)

    if cpu > 85:
        draw_text(20, WINDOW_H - 250, "⚠ CPU Overload!", color=(1, 0.2, 0.2), size=0.3)
    if gpu > 90:
        draw_text(20, WINDOW_H - 300, "⚠ GPU Overload!", color=(1, 0.2, 0.2), size=0.3)

    # Optional: latency (if server sends timestamps)
    if "server_time" in shared_state:
        latency = (time.time() - shared_state["server_time"]) * 1000.0
        draw_text(WINDOW_W - 220, WINDOW_H - 100, f"Latency: {latency:.1f} ms", color=(0.9, 0.9, 0.5), size=0.15)

    # Restore state
    glEnable(GL_LIGHTING)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_sidebar():
    # Animate slide
    if STATE.sidebar_anim_x < STATE.sidebar_target_x:
        STATE.sidebar_anim_x = min(STATE.sidebar_anim_x + STATE.sidebar_speed, STATE.sidebar_target_x)
    elif STATE.sidebar_anim_x > STATE.sidebar_target_x:
        STATE.sidebar_anim_x = max(STATE.sidebar_anim_x - STATE.sidebar_speed, STATE.sidebar_target_x)

    # If fully closed, skip
    if STATE.sidebar_anim_x <= -300:
        return

    width = 300

    # Switch to 2D projection
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, WINDOW_W, 0, WINDOW_H)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Save state
    glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT)

    # Sidebar background (semi-transparent panel only, no full-screen clear)
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glColor4f(0.08, 0.08, 0.15, 0.9)  # Dark translucent panel
    glBegin(GL_QUADS)
    glVertex2f(STATE.sidebar_anim_x, 0)
    glVertex2f(STATE.sidebar_anim_x + width, 0)
    glVertex2f(STATE.sidebar_anim_x + width, WINDOW_H)
    glVertex2f(STATE.sidebar_anim_x, WINDOW_H)
    glEnd()

    # Text helper
    def draw_text(x, y, text, color=(1, 1, 1), size=0.3):
        glColor3f(*color)
        glPushMatrix()
        glTranslatef(STATE.sidebar_anim_x + x, y, 0)
        glScalef(size, size, size)
        for ch in text:
            glutStrokeCharacter(GLUT_STROKE_ROMAN, ord(ch))
        glPopMatrix()

    # Sidebar content
    draw_text(20, WINDOW_H - 120, "== Sidebar Menu ==", color=(0.2, 0.8, 1), size=0.3)
    draw_text(20, WINDOW_H - 200, "1. Notes", color=(1, 1, 1), size=0.20)
    draw_text(20, WINDOW_H - 260, "2. Drawing", color=(1, 1, 1), size=0.20)
    draw_text(20, WINDOW_H - 320, "3. Close Sidebar", color=(1, 0.5, 0.5), size=0.20)

    if STATE.last_action_message:
        draw_text(40, 150, f"> {STATE.last_action_message}", color=(0.9, 0.9, 0.6), size=0.25)

    # Restore state
    glPopAttrib()
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)



def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_sky_background()
    glLoadIdentity()
    gluLookAt(0,0,shared_state["dist"],0,0,0,0,1,0)
    glRotatef(shared_state["rot_x"], 1, 0, 0)
    glRotatef(shared_state["rot_y"], 0, 1, 0)

    if STATE.mesh_ready and STATE.model_scene:
        glPushMatrix()
        glScalef(STATE.model_scale,STATE.model_scale,STATE.model_scale)
        glTranslatef(-STATE.model_center[0],-STATE.model_center[1],-STATE.model_center[2])
        try:
            pw_draw(STATE.model_scene)
        except Exception:
            for mesh in STATE.model_scene.mesh_list:
                for mat in mesh.materials:
                    verts = getattr(mat,'vertices',[])
                    glBegin(GL_TRIANGLES)
                    for i in range(0,len(verts),8):
                        x,y,z = verts[i:i+3]
                        nx,ny,nz = verts[i+3:i+6]
                        u,v = verts[i+6:i+8]
                        glNormal3f(nx,ny,nz)
                        glTexCoord2f(u,v)
                        glVertex3f(x,y,z)
                    glEnd()
        glPopMatrix()
    else:
        glColor3f(0.3,0.7,1.0)
        glutSolidTeapot(1.0)

    # draw overlays
    draw_sidebar()
    draw_hud()

    glutSwapBuffers()
    STATE.frame_count += 1
    now = time.time()
    if now - STATE.last_fps_time >= 0.5:
        STATE.current_fps = STATE.frame_count/(now - STATE.last_fps_time)
        STATE.frame_count = 0
        STATE.last_fps_time = now

# ---------------- camera thread ----------------
def camera_worker(device_index=0):
    cap = cv2.VideoCapture(device_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    STATE.cap = cap
    hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=2)
    STATE.hands = hands
    frame_i = 0
    try:
        while STATE.running and cap.isOpened():
            ok,frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            frame = cv2.flip(frame,1)
            small = cv2.resize(frame, CAM_DOWNSCALE, interpolation=cv2.INTER_LINEAR)
            frame_i += 1
            if frame_i % MEDIAPIPE_EVERY_N==0:
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                with STATE._lock:
                    STATE._last_processed = res
                    if SHOW_DEBUG_WINDOW:
                        STATE._last_frame = small.copy()
            else:
                if SHOW_DEBUG_WINDOW and (frame_i % (MEDIAPIPE_EVERY_N*5)==0):
                    with STATE._lock:
                        STATE._last_frame = small.copy()
            time.sleep(0.001)
    except Exception as e:
        print("Camera thread error:",e)
    finally:
        try: hands.close()
        except Exception: pass
        try: cap.release()
        except Exception: pass
        STATE.cap = None

# ---------------- gesture detection ----------------
def detect_hand_combination(lm, w, h):
    thumb = lm[mp_hands.HandLandmark.THUMB_TIP]
    index = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring = lm[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky = lm[mp_hands.HandLandmark.PINKY_TIP]

    tx, ty = thumb.x * w, thumb.y * h
    ix, iy = index.x * w, index.y * h
    mx, my = middle.x * w, middle.y * h
    rx, ry = ring.x * w, ring.y * h
    px, py = pinky.x * w, pinky.y * h

    d_index = math.hypot(tx - ix, ty - iy)
    d_middle = math.hypot(tx - mx, ty - my)
    d_ring = math.hypot(tx - rx, ty - ry)
    d_pinky = math.hypot(tx - px, ty - py)

    threshold = PINCH_PIXEL_THRESHOLD

    # Directional (rotation/tilt)
    if d_index < threshold and d_middle > threshold and d_ring > threshold and d_pinky > threshold:
        return "LEFT"
    if d_middle < threshold and d_index > threshold and d_ring > threshold and d_pinky > threshold:
        return "RIGHT"
    if d_ring < threshold and d_index > threshold and d_middle > threshold and d_pinky > threshold:
        return "UP"
    if d_pinky < threshold and d_index > threshold and d_middle > threshold and d_ring > threshold:
        return "DOWN"

    # Zoom gestures (new)
    if d_index < threshold and d_middle < threshold and d_ring > threshold and d_pinky > threshold:
        return "ZOOM_IN"
    if d_index < threshold and d_middle < threshold and d_ring < threshold and d_pinky > threshold:
        return "ZOOM_OUT"

    return None


def is_finger_extended(landmarks, tip_idx, pip_idx):
    # In Mediapipe coordinates: y increases downward.
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    # if tip.y is smaller than pip.y -> finger is up (extended)
    return tip.y < pip.y

def count_extended_fingers(landmarks):
    # landmarks is a list of 21 Landmark objects
    # We'll check index, middle, ring, pinky by tip vs pip
    extended = 0
    try:
        if is_finger_extended(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP):
            extended += 1
        if is_finger_extended(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP):
            extended += 1
        if is_finger_extended(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP):
            extended += 1
        if is_finger_extended(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP):
            extended += 1
        # Thumb: check horizontal difference between tip and ip joint as approximate
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
        # If thumb tip is sufficiently away from thumb_ip (x axis), treat as extended
        if abs(thumb_tip.x - thumb_ip.x) > 0.03:
            extended += 1
    except Exception:
        pass
    return extended

def start_note_recording_thread(note_file_path):
    """Spawn a thread to capture audio and transcribe it to text notes."""
    def worker():
        STATE.is_recording_note = True
        STATE.last_action_message = "Starting note capture..."
        speak("Starting note capture")
        text = ""

        if SR_AVAILABLE:
            try:
                r = sr.Recognizer()
                with sr.Microphone() as mic:
                    r.adjust_for_ambient_noise(mic)
                    audio = r.listen(mic, timeout=10, phrase_time_limit=30)
                text = r.recognize_google(audio)
            except Exception as e:
                text = f"[Error capturing note: {e}]"
        else:
            text = "[SpeechRecognition not installed]"

        try:
            with open(note_file_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
            STATE.last_action_message = "Note saved!"
            speak("Note saved")
        except Exception as e:
            STATE.last_action_message = f"Failed to save note: {e}"
            speak("Failed to save note")

        STATE.is_recording_note = False

    threading.Thread(target=worker, daemon=True).start()

def update_from_last_hand():
    with STATE._lock:
        res = STATE._last_processed
    if res is None or not getattr(res, 'multi_hand_landmarks', None):
        STATE.current_gesture = "None"
        return

    hl = res.multi_hand_landmarks[0]
    extended_count = count_extended_fingers(hl.landmark)

    # Sidebar open/close gestures: fist (0) open, 3 fingers close
    if extended_count == 0 and not STATE.sidebar_open:
        STATE.sidebar_open = True
        STATE.sidebar_target_x = 0
        STATE.last_action_message = "Sidebar opened"
        speak("Sidebar opened")
    elif extended_count == 3 and STATE.sidebar_open and STATE.sidebar_selection is None:
        STATE.sidebar_open = False
        STATE.sidebar_target_x = -300
        STATE.last_action_message = "Sidebar closed"
        speak("Sidebar closed")

    # If sidebar is open, check for selection gestures
    if STATE.sidebar_open:
        idx_extended = is_finger_extended(
            hl.landmark,
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.INDEX_FINGER_PIP
        )
        mid_extended = is_finger_extended(
            hl.landmark,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP
        )
        ring_extended = is_finger_extended(
            hl.landmark,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_PIP
        )

        # 1 finger -> Notes
        if idx_extended and not mid_extended and not ring_extended:
            if STATE.sidebar_selection != 1 and not STATE.is_recording_note:
                STATE.sidebar_selection = 1
                STATE.last_action_message = "Selected Notes (1). Recording..."
                speak("Notes selected. Recording.")
                note_path = os.path.join(NOTES_DIR, "1.txt")
                start_note_recording_thread(note_path)

                def clear_sel_after_delay():
                    time.sleep(1.0)
                    STATE.sidebar_selection = None
                threading.Thread(target=clear_sel_after_delay, daemon=True).start()

        # 2 fingers -> Drawing
        elif idx_extended and mid_extended and not ring_extended:
            if STATE.sidebar_selection != 2:
                STATE.sidebar_selection = 2
                STATE.last_action_message = "Selected Drawing (2)."
                speak("Drawing selected")

                def clear_sel_after_delay():
                    time.sleep(1.0)
                    STATE.sidebar_selection = None
                threading.Thread(target=clear_sel_after_delay, daemon=True).start()

        # 3 fingers -> Close Sidebar
        elif idx_extended and mid_extended and ring_extended:
            if STATE.sidebar_selection != 3:
                STATE.sidebar_selection = 3
                STATE.last_action_message = "Selected Close Sidebar (3). Closing sidebar..."
                speak("Closing sidebar")
                STATE.sidebar_open = False
                STATE.sidebar_target_x = -300

                def clear_sel_and_message():
                    time.sleep(0.6)
                    STATE.sidebar_selection = None
                    STATE.last_action_message = ""
                threading.Thread(target=clear_sel_and_message, daemon=True).start()
        else:
            pass
        updated = False
        prev_state = shared_state.copy()

        # Example for left/right/up/down rotation gestures:
        if gesture == "LEFT":
            shared_state["rot_y"] -= rotate_amount
            updated = True
        elif gesture == "RIGHT":
            shared_state["rot_y"] += rotate_amount
            updated = True
        elif gesture == "UP":
            shared_state["rot_x"] -= rotate_amount
            updated = True
        elif gesture == "DOWN":
            shared_state["rot_x"] += rotate_amount
            updated = True

        if updated:
            broadcast_state(throttle=0)
        STATE.current_gesture = f"{extended_count}fingers (sidebar)"
        return

    # If sidebar closed, allow gestures to move/zoom
    gesture = detect_hand_combination(hl.landmark, *CAM_DOWNSCALE)
    STATE.current_gesture = gesture or f"{extended_count}fingers"
    rotate_amount = 2.0
    zoom_amount = 0.2

    updated = False
    prev_state = shared_state.copy()

    updated = False
    if gesture == "LEFT":
        shared_state["rot_y"] -= rotate_amount
        updated = True
    elif gesture == "RIGHT":
        shared_state["rot_y"] += rotate_amount
        updated = True
    elif gesture == "UP":
        shared_state["rot_x"] -= rotate_amount
        updated = True
    elif gesture == "DOWN":
        shared_state["rot_x"] += rotate_amount
        updated = True
    elif gesture == "ZOOM_IN":
        shared_state["dist"] = clamp(shared_state["dist"] - zoom_amount, 1.0, 40.0)
        updated = True
    elif gesture == "ZOOM_OUT":
        shared_state["dist"] = clamp(shared_state["dist"] + zoom_amount, 1.0, 40.0)
        updated = True

    if updated:
        alerts = run_safety_checks(prev_state, shared_state)
        shared_state["alerts"] = alerts
        if alerts:
            STATE.last_action_message = alerts[0]
            speak(alerts[0])
        broadcast_state()


# ---------------- GLUT timer ----------------
def timer_func(value):
    update_from_last_hand()
    glutPostRedisplay()
    glutTimerFunc(int(1000.0 / TARGET_FPS), timer_func, 0)

# ---------------- input handlers ----------------
def keyboard(key,x,y):
    k = key.decode('utf-8') if isinstance(key,bytes) else str(key)
    if k.lower()=='w': STATE.wireframe = not STATE.wireframe
    elif k.lower()=='b': STATE.bg_mode = 1-STATE.bg_mode
    elif k.lower()=='r':
        STATE.view = ViewState()
    elif k=='\x1b': shutdown_and_exit()
    elif k.lower()=='s':
        # manual toggle sidebar
        STATE.sidebar_open = not STATE.sidebar_open
        STATE.sidebar_target_x = 0 if STATE.sidebar_open else -300
        STATE.last_action_message = "Sidebar toggled (manual)"
        speak("Sidebar toggled")

# ---------------- shutdown ----------------
def shutdown_and_exit():
    msg = "[Exit] shutting down..."
    print(msg)
    speak("Shutting down")
    STATE.running = False
    time.sleep(0.05)
    try:
        if STATE.cap is not None:
            STATE.cap.release()
    except Exception:
        pass
    try: cv2.destroyAllWindows()
    except Exception: pass
    try: sys.exit(0)
    except SystemExit:
        os._exit(0)

atexit.register(lambda:setattr(STATE,'running',False))

# ---------------- Tkinter launcher ----------------
class LauncherUI:
    def __init__(self, master):
        self.master = master
        master.title("Hand-Tracked 3D Viewer")
        master.geometry("540x360")
        frm = ttk.Frame(master, padding=12)
        frm.pack(fill='both', expand=True)

        self.model_label = tk.StringVar(value="No model selected")
        ttk.Label(frm, textvariable=self.model_label).pack(anchor='w')
        ttk.Button(frm, text="Load OBJ", command=self.choose_model).pack(fill='x', pady=4)

        self.mtl_label = tk.StringVar(value="No MTL selected For Transfering")
        ttk.Label(frm, textvariable=self.mtl_label).pack(anchor='w')
        ttk.Button(frm, text="Send Extra", command=self.choose_extra).pack(fill='x', pady=4)

        self.bgimg_label = tk.StringVar(value="No background image")
        ttk.Label(frm, textvariable=self.bgimg_label).pack(anchor='w')
        ttk.Button(frm, text="Choose Background", command=self.choose_bg_image).pack(fill='x', pady=4)

        ttk.Label(frm, text="Notes folder: " + NOTES_DIR).pack(anchor='w', pady=(6,0))
        ttk.Button(frm, text="Start Viewer", command=self.start_viewer).pack(fill='x', pady=8)
        ttk.Separator(frm).pack(fill='x', pady=6)
        self.proc_every = tk.IntVar(value=MEDIAPIPE_EVERY_N)

    def choose_extra(self):
        """Prompt user to select one or more 'extra' files (any type) for transfer"""
        paths = filedialog.askopenfilenames(title="Select Extra Files", filetypes=[("All Files", "*.*")])
        if paths:
            self.selected_extra_files = list(paths)
            self.mtl_label.set(f"{len(paths)} file(s) selected for transfer.")
        else:
            self.selected_extra_files = []
            self.mtl_label.set("No Extra selected For Transfering")



    def choose_model(self):
        path=filedialog.askopenfilename(title="Select OBJ",filetypes=[("OBJ","*.obj")])
        if path: 
            self.model_path=path
            self.model_label.set(os.path.basename(path))

    def choose_bg_image(self):
        path=filedialog.askopenfilename(title="Select Background Image", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            STATE.bg_image_path=path
            self.bgimg_label.set(os.path.basename(path))

    def start_viewer(self):
        global MEDIAPIPE_EVERY_N, SHOW_DEBUG_WINDOW
        try: MEDIAPIPE_EVERY_N=max(1,int(self.proc_every.get()))
        except: MEDIAPIPE_EVERY_N=2
        # SHOW_DEBUG_WINDOW=bool(self.debug_win.get())
        if self.model_path:
            load_obj(self.model_path)
            send_model_to_clients(self.model_path)
        threading.Thread(target=camera_worker,daemon=True).start()
        self.master.destroy()
        run_glut()

        
def run_glut():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(WINDOW_W, WINDOW_H)
    glutCreateWindow(b"Hand-Tracked 3D Viewer")

    init_gl(WINDOW_W, WINDOW_H)

    # Callbacks
    glutDisplayFunc(display)
    glutReshapeFunc(resize_gl)
    glutKeyboardFunc(keyboard)
    glutTimerFunc(int(1000.0 / TARGET_FPS), timer_func, 0)

    # Enter main loop
    glutMainLoop()
if __name__ == "__main__":
    root = tk.Tk()
    ui = LauncherUI(root)
    root.mainloop()
