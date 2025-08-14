import os
import cv2
import time
import math
import json
import queue
import threading
import platform
from collections import defaultdict
from typing import List, Dict, Any

from flask import Flask, Response, send_from_directory, jsonify, request

import mediapipe as mp
import numpy as np
import rtmidi

# =========================
# Config
# =========================
CAM_WIDTH = 1280
CAM_HEIGHT = 720
# CAM_WIDTH = 960  # Uncomment for lower resolution
# CAM_HEIGHT = 540  # Uncomment for lower resolution
CAM_INDEX = 0

# MIDI mapping
OPEN_PALM_NOTE = 64
NOTE_VELOCITY = 100
CC_WRIST_ROTATION = 3
CC_PINCH = 4
CC_Y_AXIS = 5

CC_DEADBAND = 2
ROTATION_CLAMP_DEG = (13, 165)
PINCH_MIN = 0.03
PINCH_MAX = 0.20

LEFT_HAND_CHANNEL = 0
RIGHT_HAND_CHANNEL = 1

MODE_1 = 1
MODE_2 = 2
MODE_NAMES = {MODE_1: "OpenPalm + Wrist", MODE_2: "Pinch + Y-axis"}

# =========================
# Helpers
# =========================
def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def norm_to_cc(value01):
    return int(clamp(value01, 0.0, 1.0) * 127)

def map_deg_to_cc(angle_deg, deg_min=0, deg_max=180):
    angle_deg = clamp(angle_deg, deg_min, deg_max)
    return int((angle_deg - deg_min) / (deg_max - deg_min) * 127)

# =========================
# Gesture Processor
# =========================
class GestureProcessor:
    def __init__(self):
        self.mode = MODE_1
        self.mapping_mode = False
        self.active_param_idx = 0
        self.prev_cc = defaultdict(lambda: -1)
        self.note_playing = {'Left': False, 'Right': False}
        # Initialize last_data so frontend has a stable structure before first frame
        self.last_data = {
            'timestamp': time.time(),
            'mode': self.mode,
            'mode_name': MODE_NAMES[self.mode],
            'mapping_mode': self.mapping_mode,
            'active_param_idx': self.active_param_idx,
            'hands': []
        }
        self.lock = threading.Lock()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=0
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.midi_out = rtmidi.MidiOut()
        ports = self.midi_out.get_ports()
        if not ports:
            self.midi_out.open_virtual_port("Gesture_Control_Backend")
        else:
            self.midi_out.open_port(0)

    def param_active(self, param_code):
        if not self.mapping_mode or self.active_param_idx == 0:
            return True
        if self.mode == MODE_1:
            mapping = {3: "M1_ROT", 4: "M1_BTN"}
        else:
            mapping = {3: "M2_PINCH", 4: "M2_Y"}
        return mapping.get(self.active_param_idx) == param_code

    def process(self, frame, ts):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        data = {
            'timestamp': ts,
            'mode': self.mode,
            'mode_name': MODE_NAMES[self.mode],
            'mapping_mode': self.mapping_mode,
            'active_param_idx': self.active_param_idx,
            'hands': []
        }
        handedness_labels = []
        if results.multi_handedness:
            for hnd in results.multi_handedness:
                handedness_labels.append(hnd.classification[0].label)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = handedness_labels[idx] if idx < len(handedness_labels) else 'Right'
                channel = LEFT_HAND_CHANNEL if label == 'Left' else RIGHT_HAND_CHANNEL
                lm = hand_landmarks.landmark
                pinky = lm[self.mp_hands.HandLandmark.PINKY_TIP]
                index_tip = lm[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                hand_info = {
                    'label': label,
                    'channel': channel,
                    'cc': {},
                    'note_on': self.note_playing[label]
                }
                if self.mode == MODE_1:
                    fingers_extended = [
                        lm[self.mp_hands.HandLandmark.THUMB_TIP].y < lm[self.mp_hands.HandLandmark.THUMB_IP].y,
                        lm[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                        lm[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                        lm[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[self.mp_hands.HandLandmark.RING_FINGER_PIP].y,
                        lm[self.mp_hands.HandLandmark.PINKY_TIP].y < lm[self.mp_hands.HandLandmark.PINKY_PIP].y,
                    ]
                    open_now = all(fingers_extended)
                    hand_info['palm_open'] = bool(open_now)
                    # Draw landmarks with color depending on palm state
                    conn_color = (180, 180, 180) if not open_now else (0, 255, 120)
                    lm_color = (200, 200, 200) if not open_now else (0, 255, 120)
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=lm_color, thickness=2, circle_radius=2),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=conn_color, thickness=2)
                    )
                    if open_now:
                        if self.param_active("M1_BTN") and not self.note_playing[label]:
                            self.midi_out.send_message([0x90 + channel, OPEN_PALM_NOTE, NOTE_VELOCITY])
                            self.note_playing[label] = True
                    else:
                        if self.note_playing[label] and self.param_active("M1_BTN"):
                            self.midi_out.send_message([0x80 + channel, OPEN_PALM_NOTE, 0])
                        self.note_playing[label] = False
                    hand_info['note_on'] = self.note_playing[label]

                    # Wrist roll
                    wrist_lm = lm[self.mp_hands.HandLandmark.WRIST]
                    index_mcp = lm[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    pinky_mcp = lm[self.mp_hands.HandLandmark.PINKY_MCP]
                    wrist_np = np.array([wrist_lm.x, wrist_lm.y, wrist_lm.z], dtype=np.float32)
                    index_np = np.array([index_mcp.x, index_mcp.y, index_mcp.z], dtype=np.float32)
                    pinky_np = np.array([pinky_mcp.x, pinky_mcp.y, pinky_mcp.z], dtype=np.float32)
                    vec1 = index_np - wrist_np
                    vec2 = pinky_np - wrist_np
                    normal = np.cross(vec1, vec2)
                    norm = np.linalg.norm(normal)
                    angle_deg = None
                    if norm > 1e-6:
                        normal /= norm
                        cam_dir = np.array([0,0,-1], dtype=np.float32)
                        dot = float(np.dot(normal, cam_dir))
                        dot = max(-1.0, min(1.0, dot))
                        angle_deg = math.degrees(math.acos(dot))
                        cc_val_raw = map_deg_to_cc(angle_deg, *ROTATION_CLAMP_DEG)
                        # Invert rotation for right hand only
                        cc_val = (127 - cc_val_raw) if label == 'Right' else cc_val_raw
                        if self.param_active("M1_ROT"):
                            key = (label, CC_WRIST_ROTATION)
                            if abs(cc_val - self.prev_cc[key]) >= CC_DEADBAND:
                                self.midi_out.send_message([0xB0 + channel, CC_WRIST_ROTATION, cc_val])
                                self.prev_cc[key] = cc_val
                        hand_info['cc']['wrist_rotation'] = cc_val
                        hand_info['wrist_angle_deg'] = angle_deg
                else:  # MODE_2
                    thumb_tip = lm[self.mp_hands.HandLandmark.THUMB_TIP]
                    dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
                    norm_pinch = clamp((dist - PINCH_MIN) / (PINCH_MAX - PINCH_MIN), 0.0, 1.0)
                    cc_pinch = norm_to_cc(norm_pinch)
                    if self.param_active("M2_PINCH"):
                        key = (label, CC_PINCH)
                        if abs(cc_pinch - self.prev_cc[key]) >= CC_DEADBAND:
                            self.midi_out.send_message([0xB0 + channel, CC_PINCH, cc_pinch])
                            self.prev_cc[key] = cc_pinch
                    hand_info['cc']['pinch'] = cc_pinch
                    hand_info['pinch_distance'] = dist
                    hand_info['pinch_norm'] = norm_pinch
                    # Draw landmarks in neutral color
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(200,200,200), thickness=2, circle_radius=2),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(180,180,180), thickness=2)
                    )
                    # Draw line between thumb tip and index tip
                    h, w = frame.shape[:2]
                    p1 = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                    p2 = (int(index_tip.x * w), int(index_tip.y * h))
                    cv2.line(frame, p1, p2, (255, 100, 100), 2)

                    cc_y = int((1.0 - pinky.y) * 127)
                    cc_y = clamp(cc_y, 0, 127)
                    if self.param_active("M2_Y"):
                        key_y = (label, CC_Y_AXIS)
                        if abs(cc_y - self.prev_cc[key_y]) >= CC_DEADBAND:
                            self.midi_out.send_message([0xB0 + channel, CC_Y_AXIS, cc_y])
                            self.prev_cc[key_y] = cc_y
                    hand_info['cc']['y_axis'] = cc_y

                data['hands'].append(hand_info)

        with self.lock:
            self.last_data = data
        return data

    def update_state(self, **kwargs):
        with self.lock:
            if 'mode' in kwargs and kwargs['mode'] in MODE_NAMES:
                self.mode = kwargs['mode']
            if 'mapping_mode' in kwargs:
                self.mapping_mode = bool(kwargs['mapping_mode'])
            if 'active_param_idx' in kwargs:
                try:
                    self.active_param_idx = int(kwargs['active_param_idx'])
                except (ValueError, TypeError):
                    pass
            if 'reset_notes' in kwargs and kwargs['reset_notes']:
                for k in self.note_playing:
                    if self.note_playing[k]:
                        # send note off for safety
                        ch = LEFT_HAND_CHANNEL if k == 'Left' else RIGHT_HAND_CHANNEL
                        self.midi_out.send_message([0x80 + ch, OPEN_PALM_NOTE, 0])
                        self.note_playing[k] = False

# =========================
# Capture Thread
# =========================
class CaptureThread(threading.Thread):
    def __init__(self, processor: GestureProcessor):
        super().__init__(daemon=True)
        self.processor = processor
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else 0
        self.cap = cv2.VideoCapture(CAM_INDEX, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.running = True
        self.frame = None
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)
            ts = time.time()
            self.processor.process(frame, ts)
            # Draw minimal overlay (hand count + mode)
            with self.lock:
                self.frame = frame
        self.cap.release()

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False

# =========================
# Flask App
# =========================
# Use /static for assets so /api/* routes are not shadowed by root-level static rule
app = Flask(__name__, static_folder='web', static_url_path='/static')
processor = GestureProcessor()
capture = CaptureThread(processor)
capture.start()

@app.route('/')
def index():
    return send_from_directory('web', 'index.html')

@app.route('/api/gesture')
def api_gesture():
    with processor.lock:
        return jsonify(processor.last_data)

@app.route('/api/state', methods=['POST'])
def api_state():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        processor.update_state(**payload)
        with processor.lock:
            return jsonify({'ok': True, 'state': processor.last_data})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400

@app.route('/stream')
def stream():
    def gen():
        while True:
            frame = capture.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            # annotate basics
            # cv2.putText(frame, MODE_NAMES[processor.mode], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,120), 2, cv2.LINE_AA)
            # hands_count = len(processor.last_data.get('hands', [])) if processor.last_data else 0
            # cv2.putText(frame, f"Hands:{hands_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2, cv2.LINE_AA)
            ret, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# =========================
# Graceful shutdown
# =========================
import atexit
@atexit.register
def shutdown():
    capture.stop()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)
