# app.py
import os, io, csv, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import streamlit as st
from ultralytics import YOLO

# -------------------- CONFIG --------------------
DEFAULT_MODEL_PATH = "best.pt"     # put your weights here (or set MODEL_URL to auto-download)
IMGSZ = 320
DEFAULT_TOPK = 5
AUDIT_LOG = "audit_log.csv"

st.set_page_config(page_title="Casting QA", page_icon="🧪", layout="wide")

# -------------------- OPTIONAL: download model --------------------
def maybe_download_model(target_path: str) -> None:
    """Download weights if MODEL_URL provided via env or Streamlit secrets."""
    url = st.secrets.get("MODEL_URL", os.getenv("MODEL_URL", "")).strip() if hasattr(st, "secrets") else os.getenv("MODEL_URL", "").strip()
    if not url or Path(target_path).exists():
        return
    import urllib.request
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    st.info("Downloading model weights…")
    urllib.request.urlretrieve(url, target_path)

# -------------------- UTILITIES --------------------
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = YOLO(path)
    model.model.eval()
    return model

def to_pil(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode != "RGB":
        img = ImageOps.grayscale(img).convert("RGB")
    return img

def pil_to_tensor(img: Image.Image, imgsz: int) -> torch.Tensor:
    img = img.resize((imgsz, imgsz))
    arr = (np.asarray(img).astype(np.float32) / 255.0).transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0)

def last_conv_layer(m: torch.nn.Module):
    last = None
    for x in m.modules():
        if isinstance(x, torch.nn.Conv2d):
            last = x
    return last

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", DEFAULT_MODEL_PATH)
    topk = st.slider("Top-K to display", 1, 5, DEFAULT_TOPK, 1)
    device_choice = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    st.caption("Tip: Set a secret **MODEL_URL** to auto-download weights at startup.")

# Try downloading if needed
maybe_download_model(model_path)

# Load model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# pick device
if device_choice == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = device_choice
model.to(device)
class_names = list(model.names.values()) if isinstance(model.names, dict) else model.names

st.title("Casting Defect Classification")
st.caption("Upload a casting image to classify as OK/DEFECT.")

uploads = st.file_uploader("Upload image(s)", type=["jpg","jpeg","png","bmp","tif"], accept_multiple_files=True)
if not uploads:
    st.info("Upload one or more images to begin…")
    st.stop()

cols = st.columns(2)
with cols[0]:
    st.subheader("Inputs")
    for f in uploads:
        st.image(to_pil(f.getvalue()), caption=f.name, use_container_width=True)

with cols[1]:
    st.subheader("Predictions")
    for f in uploads:
        pil = to_pil(f.getvalue())
        start = time.time()
        # Ultralytics predict handles preprocessing internally and returns class probs
        results = model.predict(source=[np.array(pil)], imgsz=IMGSZ, device=device, verbose=False)
        elapsed_ms = (time.time() - start) * 1000

        r = results[0]
        if r.probs is None:
            st.error("No probabilities returned; ensure classification weights (yolov8*-cls).")
            continue

        probs = r.probs.data.cpu().numpy()
        idx = np.argsort(-probs)[:topk]
        names = [class_names[i] for i in idx]
        scores = [float(probs[i]) for i in idx]

        st.write(f"**{f.name}** — inference: **{elapsed_ms:.1f} ms**")
        st.metric("Top-1", names[0], f"{scores[0]*100:.2f}%")
        st.dataframe({"class": names, "confidence": [f"{s*100:.2f}%" for s in scores]}, use_container_width=True)

st.markdown("---")
st.caption("Note: This app performs **image-level classification** (e.g., def_front vs ok_front). "
           "Keep validation/test sets augmentation-free for fair evaluation.")
