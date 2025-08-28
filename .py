import cv2
import mediapipe as mp
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pywavefront
from pywavefront.visualization import draw
from tkinter import Tk, Button, filedialog, Label

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Rotation variables
rotation_x, rotation_y = 0, 0
prev_x, prev_y = None, None
rotating = False
smoothed_index_x, smoothed_index_y = 0, 0
smoothed_rotation_x, smoothed_rotation_y = 0, 0
alpha = 0.2

# OpenGL
model_scene = None
cap = cv2.VideoCapture(0)

# Tkinter GUI
root = Tk()
root.title("3D Object Loader with Textures")

Label(root, text="Select 3D Model (.obj)").pack(pady=5)
Label(root, text="Select Texture Images (optional)").pack(pady=5)

textures = {}

def load_model():
    global model_scene
    obj_path = filedialog.askopenfilename(title="Select OBJ file", filetypes=[("OBJ Files", "*.obj")])
    if obj_path:
        model_scene = pywavefront.Wavefront(obj_path, collect_faces=True, parse=True)
        info_label.config(text=f"Loaded: {obj_path.split('/')[-1]}")

def load_textures():
    global textures
    file_paths = filedialog.askopenfilenames(title="Select all texture files", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if file_paths:
        for file in file_paths:
            name = file.split("/")[-1]
            textures[name] = file
        info_label.config(text=f"Textures Loaded: {len(file_paths)} files")

Button(root, text="Load 3D Model", command=load_model).pack(pady=5)
Button(root, text="Load Textures", command=load_textures).pack(pady=5)
info_label = Label(root, text="No model or textures loaded")
info_label.pack(pady=10)

def start_opengl():
    root.destroy()
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(1080, 1920)
    glutCreateWindow(b"Hand-Controlled 3D Object")
    initOpenGL()
    glutDisplayFunc(display)
    glutIdleFunc(idle)
    glutMainLoop()

Button(root, text="Start 3D Viewer", command=start_opengl).pack(pady=20)

def initOpenGL():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_TEXTURE_2D)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 800/600, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity(),
    glTranslatef(0, 2, -20)
    glRotatef(smoothed_rotation_x, 1, 0, 0)
    glRotatef(smoothed_rotation_y, 0, 1, 0)

    if model_scene:
        # Attempt to bind textures if present
        for name, mat in model_scene.materials.items():
            if hasattr(mat, 'texture') and mat.texture is not None:
                tex_file = textures.get(mat.texture)
                if tex_file:
                    # Here we would load the texture into OpenGL (PyWavefront does simple binding if paths match)
                    mat.texture.image_path = tex_file
        draw(model_scene)
    else:
        drawCuboid()

    glutSwapBuffers()

def drawCuboid():
    vertices = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ]
    faces = [
        (0,1,2,3),(4,5,6,7),(0,1,5,4),
        (2,3,7,6),(0,3,7,4),(1,2,6,5)
    ]
    colors = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1)]
    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glColor3fv(colors[i])
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

def idle():
    global rotating, prev_x, prev_y
    global rotation_x, rotation_y
    global smoothed_index_x, smoothed_index_y
    global smoothed_rotation_x, smoothed_rotation_y

    ret, frame = cap.read()
    if not ret:
        return
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    h, w, _ = frame.shape
    rotating = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]

            index_x, index_y = int(index_tip.x*w), int(index_tip.y*h)
            smoothed_index_x += alpha * (index_x - smoothed_index_x)
            smoothed_index_y += alpha * (index_y - smoothed_index_y)

            thumb_x, thumb_y = int(thumb_tip.x*w), int(thumb_tip.y*h)
            pinch_distance = np.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
            if pinch_distance < 50:
                rotating = True

            if rotating:
                if prev_x is None:
                    prev_x, prev_y = smoothed_index_x, smoothed_index_y
                dx = smoothed_index_x - prev_x
                dy = smoothed_index_y - prev_y
                rotation_y += dx * 0.5
                rotation_x += dy * 0.5
                smoothed_rotation_x += alpha * (rotation_x - smoothed_rotation_x)
                smoothed_rotation_y += alpha * (rotation_y - smoothed_rotation_y)
                prev_x, prev_y = smoothed_index_x, smoothed_index_y
            else:
                prev_x, prev_y = None, None

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking", frame)
    cv2.waitKey(1)
    glutPostRedisplay()

root.mainloop()
