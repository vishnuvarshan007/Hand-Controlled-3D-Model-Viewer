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

# ---------------- OpenGL rendering ----------------
WINDOW_W, WINDOW_H = 1280, 720

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

def set_bg():
    if STATE.bg_mode==0: glClearColor(0.05,0.05,0.08,1.0)
    else: glClearColor(0.95,0.95,0.98,1.0)

def draw_hud():
    # top-right HUD with model info, fps, rot, dist
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WINDOW_W, WINDOW_H, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_LIGHTING)
    if STATE.bg_mode==0:
        glColor3f(1,1,1)
    else:
        glColor3f(0.08,0.08,0.08)

    def text(x,y,s):
        glRasterPos2f(x,y)
        for ch in s:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))

    # draw top-right; leave some padding
    x = WINDOW_W - 320
    y0 = 18
    text(x, y0, f"Model: {STATE.model_name}")
    text(x, y0+20, f"FPS: {STATE.current_fps:.1f}")
    text(x, y0+40, f"RotX: {STATE.view.rot_x:.1f}")
    text(x, y0+60, f"RotY: {STATE.view.rot_y:.1f}")
    text(x, y0+80, f"Dist: {STATE.view.dist:.2f}")
    text(x, y0+100, f"Gesture: {STATE.current_gesture}")

    glEnable(GL_LIGHTING)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_sidebar_overlay():
    # Animate sidebar x position
    if STATE.sidebar_anim_x < STATE.sidebar_target_x:
        STATE.sidebar_anim_x += STATE.sidebar_speed
        if STATE.sidebar_anim_x > STATE.sidebar_target_x:
            STATE.sidebar_anim_x = STATE.sidebar_target_x
    elif STATE.sidebar_anim_x > STATE.sidebar_target_x:
        STATE.sidebar_anim_x -= STATE.sidebar_speed
        if STATE.sidebar_anim_x < STATE.sidebar_target_x:
            STATE.sidebar_anim_x = STATE.sidebar_target_x

    if STATE.sidebar_anim_x <= -300:
        return

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WINDOW_W, WINDOW_H, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_LIGHTING)
    sidebar_w = 300
    if STATE.bg_mode == 0:
        glColor4f(0.06, 0.06, 0.08, 0.95)
    else:
        glColor4f(0.96, 0.96, 0.98, 0.95)
    glBegin(GL_QUADS)
    glVertex2f(STATE.sidebar_anim_x, 0)
    glVertex2f(STATE.sidebar_anim_x + sidebar_w, 0)
    glVertex2f(STATE.sidebar_anim_x + sidebar_w, WINDOW_H)
    glVertex2f(STATE.sidebar_anim_x, WINDOW_H)
    glEnd()

    if STATE.bg_mode == 0:
        glColor3f(1, 1, 1)
    else:
        glColor3f(0.08, 0.08, 0.08)

    def text(x, y, s):
        glRasterPos2f(STATE.sidebar_anim_x + x, y)
        for ch in s:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))

    text(12, 36, "Sidebar - Options:")
    text(12, 72, ( "[1]" if STATE.sidebar_selection==1 else "[ ]" ) + " 1 - Notes (speak to save)")
    text(12, 100, ( "[2]" if STATE.sidebar_selection==2 else "[ ]" ) + " 2 - Drawing Mode (placeholder)")
    text(12, 128, ( "[3]" if STATE.sidebar_selection==3 else "[ ]" ) + " 3 - Close Sidebar")
    text(12, 160, "Status:")
    if STATE.is_recording_note:
        text(12, 188, "Recording note... (listening)")
    else:
        preview = STATE.last_note_preview or "(no notes saved yet)"
        if len(preview) > 80:
            preview = preview[:77] + "..."
        text(12, 188, "Last note:")
        text(12, 206, preview)

    if STATE.last_action_message:
        text(12, WINDOW_H - 28, STATE.last_action_message)

    glEnable(GL_LIGHTING)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def display():
    set_bg()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_sky_background()
    glLoadIdentity()
    gluLookAt(0,0,STATE.view.dist,0,0,0,0,1,0)

    glRotatef(STATE.view.rot_x,1,0,0)   # pitch
    glRotatef(STATE.view.rot_y,0,1,0)   # yaw

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
    draw_sidebar_overlay()
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
    # keeps old behaviour for directional gestures (thumb touching finger tips)
    thumb = lm[mp_hands.HandLandmark.THUMB_TIP]
    index = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring = lm[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky = lm[mp_hands.HandLandmark.PINKY_TIP]
    tx, ty = thumb.x*w, thumb.y*h
    ix, iy = index.x*w, index.y*h
    mx, my = middle.x*w, middle.y*h
    rx, ry = ring.x*w, ring.y*h
    px, py = pinky.x*w, pinky.y*h
    d_index = math.hypot(tx-ix, ty-iy)
    d_middle = math.hypot(tx-mx, ty-my)
    d_ring = math.hypot(tx-rx, ty-ry)
    d_pinky = math.hypot(tx-px, ty-py)
    threshold = PINCH_PIXEL_THRESHOLD
    # Directional (rotation/tilt)
    if d_index<threshold and d_middle>threshold and d_ring>threshold and d_pinky>threshold: return "LEFT"
    if d_middle<threshold and d_index>threshold and d_ring>threshold and d_pinky>threshold: return "RIGHT"
    if d_ring<threshold and d_index>threshold and d_middle>threshold and d_pinky>threshold: return "UP"
    if d_pinky<threshold and d_index>threshold and d_middle>threshold and d_ring>threshold: return "DOWN"
    # Zoom gestures (new)
    if d_index<ththreshold if False else d_index<threshold and d_middle<ththreshold if False else d_middle<threshold and d_ring>threshold and d_pinky>threshold:
        return "ZOOM_IN"   # Thumb + Index + Middle
    if d_index < threshold and d_middle < threshold and d_ring < threshold and d_pinky > threshold:
        return "ZOOM_OUT"  # Thumb + Index + Middle + Ring close
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
    # spawn a thread to capture audio (or typed input fallback) and save to note file
    def worker():
        STATE.is_recording_note = True
        STATE.last_action_message = "Starting note capture..."
        speak("Starting note capture")
        text = ""
        if SR_AVAILABLE:
            try:
                r = sr.Recognizer()
                with sr.Microphone() as mic:
                    STATE.last_action_message = "Listening..."
                    speak("Listening")
                    # adjust for ambient noise briefly
                    r.adjust_for_ambient_noise(mic, duration=0.6)
                    audio = r.listen(mic, phrase_time_limit=10)
                STATE.last_action_message = "Recognizing..."
                speak("Recognizing")
                try:
                    text = r.recognize_google(audio)
                except Exception as e:
                    STATE.last_action_message = f"Speech recog failed: {e}"
                    text = ""
            except Exception as e:
                STATE.last_action_message = f"Microphone error: {e}"
                text = ""
        else:
            # fallback to typed input dialog (non-blocking to GLUT: use tkinter in the thread)
            try:
                root = tk.Tk()
                root.withdraw()
                text = simpledialog.askstring("Notes (fallback)", "Speech not available. Type your note:")
                try: root.destroy()
                except: pass
                if text is None:
                    text = ""
            except Exception as e:
                print("Fallback dialog error:", e)
                text = ""

        # Save text (append) with timestamp
        try:
            os.makedirs(os.path.dirname(note_file_path), exist_ok=True)
            with open(note_file_path, "a", encoding="utf-8") as f:
                if text.strip():
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{ts}] {text.strip()}\n")
                    STATE.last_note_preview = text.strip()
                    STATE.last_action_message = f"Note saved to {note_file_path}"
                    speak("Note saved")
                else:
                    STATE.last_action_message = "No note captured"
                    speak("No note captured")
        except Exception as e:
            STATE.last_action_message = f"Error saving note: {e}"
            speak("Error saving note")
        STATE.is_recording_note = False

    t = threading.Thread(target=worker, daemon=True)
    t.start()

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
        # If sidebar is open and no explicit option selected yet, treat 3-finger gesture as "close sidebar"
        STATE.sidebar_open = False
        STATE.sidebar_target_x = -300
        STATE.last_action_message = "Sidebar closed"
        speak("Sidebar closed")

    # If sidebar is open, check for selection gestures
    if STATE.sidebar_open:
        # detect index, middle, ring extended
        idx_extended = is_finger_extended(hl.landmark, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
        mid_extended = is_finger_extended(hl.landmark, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
        ring_extended = is_finger_extended(hl.landmark, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)

        # 1 finger -> Notes
        if idx_extended and not mid_extended and not ring_extended:
            if STATE.sidebar_selection != 1 and not STATE.is_recording_note:
                STATE.sidebar_selection = 1
                STATE.last_action_message = "Selected Notes (1). Recording..."
                speak("Notes selected. Recording.")
                note_path = os.path.join(NOTES_DIR, "1.txt")
                start_note_recording_thread(note_path)

                # leave the [1] showing a bit then clear selection (optional)
                def clear_sel_after_delay():
                    time.sleep(1.0)
                    STATE.sidebar_selection = None
                threading.Thread(target=clear_sel_after_delay, daemon=True).start()

        # 2 fingers (index+middle) -> Drawing (placeholder)
        elif idx_extended and mid_extended and not ring_extended:
            if STATE.sidebar_selection != 2:
                STATE.sidebar_selection = 2
                STATE.last_action_message = "Selected Drawing (2)."
                speak("Drawing selected")
                def clear_sel_after_delay():
                    time.sleep(1.0)
                    STATE.sidebar_selection = None
                threading.Thread(target=clear_sel_after_delay, daemon=True).start()

        # 3 fingers (index+middle+ring) -> Close Sidebar (NOT exit application)
        elif idx_extended and mid_extended and ring_extended:
            if STATE.sidebar_selection != 3:
                STATE.sidebar_selection = 3
                STATE.last_action_message = "Selected Close Sidebar (3). Closing sidebar..."
                speak("Closing sidebar")
                # close sidebar (do NOT exit program)
                STATE.sidebar_open = False
                STATE.sidebar_target_x = -300

                # clear selection after a moment so UI resets
                def clear_sel_and_message():
                    time.sleep(0.6)
                    STATE.sidebar_selection = None
                    STATE.last_action_message = ""
                threading.Thread(target=clear_sel_and_message, daemon=True).start()
        else:
            # no clear selection, leave selection as-is (or clear if you prefer)
            pass

        # when sidebar open we do not rotate/zoom the object
        STATE.current_gesture = f"{extended_count}fingers (sidebar)"
        return

    # If sidebar closed, allow gestures to move/zoom
    gesture = detect_hand_combination(hl.landmark, *CAM_DOWNSCALE)
    STATE.current_gesture = gesture or f"{extended_count}fingers"
    rotate_amount = 2.0
    zoom_amount = 0.2
    if gesture=="LEFT": STATE.view.rot_y -= rotate_amount
    elif gesture=="RIGHT": STATE.view.rot_y += rotate_amount
    elif gesture=="UP": STATE.view.rot_x -= rotate_amount
    elif gesture=="DOWN": STATE.view.rot_x += rotate_amount
    elif gesture=="ZOOM_IN": STATE.view.dist = clamp(STATE.view.dist - zoom_amount, 1.0, 40.0)
    elif gesture=="ZOOM_OUT": STATE.view.dist = clamp(STATE.view.dist + zoom_amount, 1.0, 40.0)

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
    def __init__(self,master):
        self.master=master
        master.title("Hand-Tracked 3D Viewer")
        master.geometry("540x360")
        frm=ttk.Frame(master,padding=12)
        frm.pack(fill='both',expand=True)
        self.model_label=tk.StringVar(value="No model selected")
        ttk.Label(frm,textvariable=self.model_label).pack(anchor='w')
        ttk.Button(frm,text="Load OBJ",command=self.choose_model).pack(fill='x',pady=4)
        # --- background image section
        self.bgimg_label=tk.StringVar(value="No background image")
        ttk.Label(frm,textvariable=self.bgimg_label).pack(anchor='w')
        ttk.Button(frm,text="Choose Background",command=self.choose_bg_image).pack(fill='x',pady=4)
        # --- notes info
        ttk.Label(frm, text="Notes folder: " + NOTES_DIR).pack(anchor='w', pady=(6,0))
        # --- 
        ttk.Button(frm,text="Start Viewer",command=self.start_viewer).pack(fill='x',pady=8)
        ttk.Separator(frm).pack(fill='x',pady=6)
        self.proc_every=tk.IntVar(value=MEDIAPIPE_EVERY_N)
        ttk.Label(frm,text="Mediapipe cadence (1 = every frame, 2 = every 2 frames):").pack(anchor='w')
        ttk.Spinbox(frm, from_=1,to=10,textvariable=self.proc_every,width=6).pack(anchor='w')
        self.debug_win=tk.BooleanVar(value=SHOW_DEBUG_WINDOW)
        ttk.Checkbutton(frm,text="Show camera debug window (slower)",variable=self.debug_win).pack(anchor='w',pady=4)
        ttk.Label(frm,text="Gestures: Fist=open sidebar, 3 fingers=close sidebar. In sidebar: 1=index, 2=index+middle, 3=index+middle+ring").pack(anchor='w',pady=(6,0))
        self.model_path = None

    def choose_model(self):
        path = filedialog.askopenfilename(title="Select OBJ",filetypes=[("OBJ","*.obj")])
        if path: 
            self.model_path=path
            self.model_label.set(os.path.basename(path))
    
    def choose_bg_image(self):
        path = filedialog.askopenfilename(title="Select Background Image", filetypes=[("Image Files","*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            STATE.bg_image_path = path
            self.bgimg_label.set(os.path.basename(path))
    
    def start_viewer(self):
        global MEDIAPIPE_EVERY_N, SHOW_DEBUG_WINDOW
        try: MEDIAPIPE_EVERY_N=max(1,int(self.proc_every.get()))
        except Exception: MEDIAPIPE_EVERY_N=2
        SHOW_DEBUG_WINDOW=bool(self.debug_win.get())
        if getattr(self,'model_path',None): load_obj(self.model_path)
        t=threading.Thread(target=camera_worker,daemon=True)
        t.start()
        self.master.destroy()
        run_glut()

# ---------------- GLUT runner ----------------
def run_glut():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH)
    glutInitWindowSize(WINDOW_W,WINDOW_H)
    glutCreateWindow(b"Hand-Tracked 3D Viewer")
    init_gl(WINDOW_W,WINDOW_H)
    # --- upload background after context, if requested
    if STATE.bg_image_path:
        load_background_image(STATE.bg_image_path)
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    glutTimerFunc(int(1000.0/TARGET_FPS), timer_func,0)
    glutMainLoop()

# ---------------- main ----------------
if __name__=="__main__":
    root=tk.Tk()
    LauncherUI(root)
    root.mainloop()
