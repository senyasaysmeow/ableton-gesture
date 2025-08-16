import os
import cv2
import time
import math
import threading
import platform
from collections import defaultdict
from typing import Dict, Any

from flask import Flask, Response, send_from_directory, jsonify, request

import mediapipe as mp
import numpy as np
import rtmidi

# =========================
# Config
# =========================
CAM_WIDTH = 960
CAM_HEIGHT = 540
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

def norm_to_cc(value01: float) -> int:
    return int(clamp(value01, 0.0, 1.0) * 127)

def map_deg_to_cc(angle_deg: float, deg_min: float = 0, deg_max: float = 180) -> int:
    angle_deg = clamp(angle_deg, deg_min, deg_max)
    if deg_max == deg_min:
        return 0
    return int((angle_deg - deg_min) / (deg_max - deg_min) * 127)

# =========================
# Gesture Processor
# =========================
class GestureProcessor:
    def __init__(self):
        # Public state
        self.mode = MODE_1
        self.mapping_mode = False
        self.active_param_idx = 0  # 0=All, else mode-specific mapping index

        # MIDI and gesture memory
        self.prev_cc = defaultdict(lambda: -1)
        self.note_playing = {'Left': False, 'Right': False}

        # Thread safety and last JSON
        self.lock = threading.Lock()
        self.last_data: Dict[str, Any] = {
            'timestamp': time.time(),
            'mode': self.mode,
            'mode_name': MODE_NAMES[self.mode],
            'mapping_mode': self.mapping_mode,
            'active_param_idx': self.active_param_idx,
            'hands': []
        }

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=0
        )

        # MIDI init
        self.midi_out = rtmidi.MidiOut()
        ports = self.midi_out.get_ports()
        try:
            if ports:
                self.midi_out.open_port(0)
            else:
                # Create a virtual port on macOS/Linux
                self.midi_out.open_virtual_port("Gesture MIDI")
        except Exception:
            # Fallback: create virtual port if opening first port failed
            try:
                self.midi_out.open_virtual_port("Gesture MIDI")
            except Exception:
                pass

    # Mapping helper: is a parameter active (respects mapping_mode and active_param_idx)
    def param_active(self, param_code: str) -> bool:
        if not self.mapping_mode or self.active_param_idx == 0:
            return True
        if self.mode == MODE_1:
            mapping = {3: 'M1_ROT', 4: 'M1_PALM'}
        else:
            mapping = {3: 'M2_PINCH', 4: 'M2_Y'}
        return mapping.get(self.active_param_idx) == param_code

    def _hand_label_and_channel(self, handedness):
        label = handedness.classification[0].label  # 'Left' or 'Right'
        channel = LEFT_HAND_CHANNEL if label == 'Left' else RIGHT_HAND_CHANNEL
        return label, channel

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

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx] if results.multi_handedness else None
                label, channel = (('Right', RIGHT_HAND_CHANNEL) if handedness is None else self._hand_label_and_channel(handedness))

                lm = hand_landmarks.landmark
                # Extract key landmarks
                wrist = lm[self.mp_hands.HandLandmark.WRIST]
                pinky = lm[self.mp_hands.HandLandmark.PINKY_TIP]
                thumb_tip = lm[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = lm[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Palm open heuristic (ignore thumb for stability)
                fingers_extended = [
                    lm[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                    lm[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                    lm[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[self.mp_hands.HandLandmark.RING_FINGER_PIP].y,
                    lm[self.mp_hands.HandLandmark.PINKY_TIP].y < lm[self.mp_hands.HandLandmark.PINKY_PIP].y,
                ]
                open_now = all(fingers_extended)

                # Wrist roll angle via palm plane normal vs camera dir
                index_mcp = lm[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
                pinky_mcp = lm[self.mp_hands.HandLandmark.PINKY_MCP]
                wrist_np = np.array([wrist.x, wrist.y, wrist.z], dtype=np.float32)
                index_np = np.array([index_mcp.x, index_mcp.y, index_mcp.z], dtype=np.float32)
                pinky_np = np.array([pinky_mcp.x, pinky_mcp.y, pinky_mcp.z], dtype=np.float32)
                vec1 = index_np - wrist_np
                vec2 = pinky_np - wrist_np
                normal = np.cross(vec1, vec2)
                norm_val = float(np.linalg.norm(normal))
                wrist_angle_deg = None
                cc_rot = None
                if norm_val > 1e-6:
                    normal /= norm_val
                    cam_dir = np.array([0, 0, -1], dtype=np.float32)
                    dot = float(np.dot(normal, cam_dir))
                    dot = max(-1.0, min(1.0, dot))
                    wrist_angle_deg = math.degrees(math.acos(dot))
                    raw_cc = map_deg_to_cc(wrist_angle_deg, *ROTATION_CLAMP_DEG)
                    # Invert rotation for right hand only
                    cc_rot = (127 - raw_cc) if label == 'Right' else raw_cc

                # Pinch
                dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
                pinch_norm = clamp((dist - PINCH_MIN) / (PINCH_MAX - PINCH_MIN), 0.0, 1.0)
                cc_pinch = norm_to_cc(pinch_norm)

                # Y-axis from wrist (top=127)
                cc_y = int((1.0 - pinky.y) * 127)
                cc_y = int(clamp(cc_y, 0, 127))

                # Prepare JSON entry
                hand_info = {
                    'label': label,
                    'palm_open': bool(open_now),
                    'wrist_angle_deg': wrist_angle_deg,
                    'pinch_distance': float(dist),
                    'pinch_norm': float(pinch_norm),
                    'cc': {
                        'wrist_rotation': int(cc_rot) if cc_rot is not None else None,
                        'pinch': int(cc_pinch),
                        'y_axis': int(cc_y),
                    }
                }

                # MIDI per mode/mapping
                if self.mode == MODE_1:
                    # Open palm note
                    want_note = open_now and self.param_active('M1_PALM')
                    if want_note and not self.note_playing[label]:
                        try:
                            self.midi_out.send_message([0x90 + channel, OPEN_PALM_NOTE, NOTE_VELOCITY])
                        except Exception:
                            pass
                        self.note_playing[label] = True
                    elif not want_note and self.note_playing[label]:
                        try:
                            self.midi_out.send_message([0x80 + channel, OPEN_PALM_NOTE, 0])
                        except Exception:
                            pass
                        self.note_playing[label] = False

                    # Wrist rotation CC
                    if cc_rot is not None and self.param_active('M1_ROT'):
                        key = (label, CC_WRIST_ROTATION)
                        if abs(cc_rot - self.prev_cc[key]) >= CC_DEADBAND:
                            try:
                                self.midi_out.send_message([0xB0 + channel, CC_WRIST_ROTATION, int(cc_rot)])
                            except Exception:
                                pass
                            self.prev_cc[key] = int(cc_rot)

                else:  # MODE_2
                    # Pinch CC
                    if self.param_active('M2_PINCH'):
                        key = (label, CC_PINCH)
                        if abs(cc_pinch - self.prev_cc[key]) >= CC_DEADBAND:
                            try:
                                self.midi_out.send_message([0xB0 + channel, CC_PINCH, int(cc_pinch)])
                            except Exception:
                                pass
                            self.prev_cc[key] = int(cc_pinch)

                    # Y-axis CC
                    if self.param_active('M2_Y'):
                        key_y = (label, CC_Y_AXIS)
                        if abs(cc_y - self.prev_cc[key_y]) >= CC_DEADBAND:
                            try:
                                self.midi_out.send_message([0xB0 + channel, CC_Y_AXIS, int(cc_y)])
                            except Exception:
                                pass
                            self.prev_cc[key_y] = int(cc_y)

                # Drawing
                if self.mode == MODE_1:
                    conn_color = (0, 255, 120) if open_now else (180, 180, 180)
                else:
                    conn_color = (180, 180, 180)
                lm_color = (200, 200, 200)
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=lm_color, thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=conn_color, thickness=2)
                )

                if self.mode == MODE_2:
                    h, w = frame.shape[:2]
                    p1 = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                    p2 = (int(index_tip.x * w), int(index_tip.y * h))
                    cv2.line(frame, p1, p2, (255, 100, 100), 2)

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
