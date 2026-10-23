from __future__ import annotations

import asyncio
import math
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "fit_orbit_bilstm.keras"
SCALER_PATH = PROJECT_DIR / "scaler.pkl"
ENCODER_PATH = PROJECT_DIR / "label_encoder.pkl"
POSE_MODEL_PATH = PROJECT_DIR / "models" / "pose_landmarker_lite.task"

WINDOW_SIZE = 30
MIN_VISIBILITY = 0.5

# Keep these names and their order identical to extract_paper_features.py.
LANDMARKS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
    "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20,
    "left_thumb": 21, "right_thumb": 22,
}

ANGLE_TRIPLETS = [
    ("left_hip", "left_shoulder", "left_elbow"),
    ("right_hip", "right_shoulder", "right_elbow"),
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
    ("left_shoulder", "left_hip", "left_knee"),
    ("right_shoulder", "right_hip", "right_knee"),
    ("left_knee", "left_ankle", "left_heel"),
    ("right_knee", "right_ankle", "right_heel"),
    ("left_ankle", "left_heel", "left_foot_index"),
    ("right_ankle", "right_heel", "right_foot_index"),
]

ANGLE_NAMES = ["angle_" + "_".join(triple) for triple in ANGLE_TRIPLETS]
FEATURE_NAMES = [
    f"{name}_{axis}"
    for name in LANDMARKS
    for axis in ("x", "y", "z")
] + ANGLE_NAMES

REQUIRED_BY_LABEL = {
    "barbell biceps curl": ("shoulder", "elbow", "wrist"),
    "push-up": ("shoulder", "elbow", "wrist", "hip", "ankle"),
    "shoulder press": ("shoulder", "elbow", "wrist"),
    "squat": ("shoulder", "hip", "knee", "ankle"),
}

app = FastAPI(title="FitOrbit Live Exercise API", version="1.0.0")
model = None
scaler = None
encoder = None


def load_artifacts():
    global model, scaler, encoder
    if model is None:
        for path in (MODEL_PATH, SCALER_PATH, ENCODER_PATH, POSE_MODEL_PATH):
            if not path.is_file():
                raise FileNotFoundError(f"Required project file not found: {path}")
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        if len(FEATURE_NAMES) != scaler.n_features_in_:
            raise ValueError(
                f"API builds {len(FEATURE_NAMES)} features, but scaler expects "
                f"{scaler.n_features_in_}. Check feature names/order."
            )
        if model.input_shape[1:] != (WINDOW_SIZE, len(FEATURE_NAMES)):
            raise ValueError(f"Unexpected model input shape: {model.input_shape}")


def visible(point) -> bool:
    return getattr(point, "visibility", 0.0) >= MIN_VISIBILITY


def has_valid_pose(points) -> bool:
    # A live frame has no known label, so accept it if it meets the training
    # extractor's validity rule for at least one of the four exercise classes.
    for joints in REQUIRED_BY_LABEL.values():
        for side in ("left", "right"):
            if all(visible(points[LANDMARKS[f"{side}_{joint}"]]) for joint in joints):
                return True
    return False


def angle_degrees(a, b, c) -> float:
    ba = [a[i] - b[i] for i in range(3)]
    bc = [c[i] - b[i] for i in range(3)]
    norm_ba = math.sqrt(sum(v * v for v in ba))
    norm_bc = math.sqrt(sum(v * v for v in bc))
    if norm_ba == 0.0 or norm_bc == 0.0:
        return 0.0
    cosine = sum(ba[i] * bc[i] for i in range(3)) / (norm_ba * norm_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def extract_features(points):
    if not has_valid_pose(points):
        return None

    coordinates = {}
    for name, index in LANDMARKS.items():
        point = points[index]
        if visible(point):
            coordinates[name] = (float(point.x), float(point.y), float(point.z))
        else:
            coordinates[name] = (0.0, 0.0, 0.0)

    row = []
    for name in LANDMARKS:
        row.extend(coordinates[name])

    for triple in ANGLE_TRIPLETS:
        if all(visible(points[LANDMARKS[name]]) for name in triple):
            row.append(angle_degrees(*(coordinates[name] for name in triple)))
        else:
            row.append(0.0)

    return np.asarray(row, dtype=np.float32)


@app.get("/health")
def health():
    try:
        load_artifacts()
        return {"status": "ready", "classes": [str(c) for c in encoder.classes_]}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.get("/", response_class=HTMLResponse)
def camera_demo():
    return HTMLResponse(CAMERA_PAGE)


@app.websocket("/ws/live")
async def live_camera(websocket: WebSocket):
    await websocket.accept()
    try:
        load_artifacts()
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1011)
        return

    frame_buffer = deque(maxlen=WINDOW_SIZE)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )

    try:
        # A separate tracker per connection keeps timestamps and tracking state isolated.
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_number = 0
            while True:
                encoded = await websocket.receive_bytes()
                if len(encoded) > 8_000_000:
                    await websocket.send_json({"type": "error", "message": "Frame is too large."})
                    continue

                image_bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    await websocket.send_json({"type": "error", "message": "Invalid image frame."})
                    continue

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                timestamp_ms = frame_number * 100  # demo sends about 10 frames/second
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                frame_number += 1

                if not result.pose_landmarks:
                    await websocket.send_json({"type": "status", "message": "No pose detected", "frames_ready": len(frame_buffer)})
                    continue

                feature_row = extract_features(result.pose_landmarks[0])
                if feature_row is None:
                    await websocket.send_json({"type": "status", "message": "Move fully into view", "frames_ready": len(frame_buffer)})
                    continue

                frame_buffer.append(feature_row)
                if len(frame_buffer) < WINDOW_SIZE:
                    await websocket.send_json({
                        "type": "status",
                        "message": f"Collecting frames: {len(frame_buffer)}/{WINDOW_SIZE}",
                        "frames_ready": len(frame_buffer),
                    })
                    continue

                window = np.asarray(frame_buffer, dtype=np.float32)[None, ...]
                scaled = scaler.transform(window.reshape(-1, window.shape[-1]))
                scaled = scaled.reshape(window.shape).astype(np.float32)
                probabilities = model.predict(scaled, verbose=0)[0]
                ranked = np.argsort(probabilities)[::-1]
                best = int(ranked[0])

                await websocket.send_json({
                    "type": "prediction",
                    "label": str(encoder.classes_[best]),
                    "confidence": float(probabilities[best]),
                    "probabilities": {
                        str(encoder.classes_[i]): float(probabilities[i])
                        for i in ranked
                    },
                    "frames_ready": len(frame_buffer),
                })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


CAMERA_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FitOrbit Live Camera</title>
<style>
body{font:16px system-ui;background:#10151d;color:#f5f7fa;max-width:850px;margin:30px auto;padding:0 16px}
button{padding:10px 18px;margin:8px 8px 8px 0;border:0;border-radius:8px;background:#64d6a2;font-weight:700;cursor:pointer}
video{width:min(100%,720px);background:#000;border-radius:12px;transform:scaleX(-1)}
#status{font-size:1.2rem;margin:12px 0}.muted{color:#abb7c4}pre{white-space:pre-wrap}
</style></head><body>
<h1>FitOrbit Live Exercise Test</h1>
<p class="muted">Allow camera access, then keep your full body visible. Prediction starts after 30 valid pose frames.</p>
<video id="video" autoplay playsinline muted></video><br>
<button id="start">Start camera</button><button id="stop" disabled>Stop</button>
<div id="status">Not connected</div><pre id="scores"></pre>
<canvas id="canvas" width="640" height="480" hidden></canvas>
<script>
const video=document.getElementById('video'), canvas=document.getElementById('canvas');
const ctx=canvas.getContext('2d'), statusEl=document.getElementById('status'), scores=document.getElementById('scores');
let stream=null, socket=null, timer=null, busy=false;
document.getElementById('start').onclick=async()=>{
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:false});
    video.srcObject=stream;
    socket=new WebSocket(`ws://${location.host}/ws/live`);
    socket.binaryType='arraybuffer';
    socket.onopen=()=>{statusEl.textContent='Camera connected. Finding pose…';document.getElementById('stop').disabled=false;sendNext();};
    socket.onmessage=e=>{const d=JSON.parse(e.data);if(d.type==='prediction'){statusEl.textContent=`${d.label} (${(d.confidence*100).toFixed(1)}%)`;scores.textContent=Object.entries(d.probabilities).map(([k,v])=>`${k}: ${(v*100).toFixed(1)}%`).join('\n');}else{statusEl.textContent=d.message||d.detail||d.type;}};
    socket.onerror=()=>statusEl.textContent='WebSocket connection error';
    socket.onclose=()=>{if(!stream)statusEl.textContent='Disconnected';};
  }catch(e){statusEl.textContent='Could not start camera: '+e.message;}
};
function sendNext(){
  if(!stream||!socket||socket.readyState!==WebSocket.OPEN)return;
  if(!busy){busy=true;ctx.drawImage(video,0,0,canvas.width,canvas.height);canvas.toBlob(async blob=>{if(blob&&socket.readyState===WebSocket.OPEN)socket.send(await blob.arrayBuffer());busy=false;},'image/jpeg',0.75);}
  timer=setTimeout(sendNext,100);
}
function stop(){clearTimeout(timer);if(socket){socket.close();socket=null;}if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}video.srcObject=null;document.getElementById('stop').disabled=true;statusEl.textContent='Stopped';}
document.getElementById('stop').onclick=stop;
window.onbeforeunload=stop;
</script></body></html>"""
