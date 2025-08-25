# --- Minimal YOLOv8 Image Inference App (images only) ---

import io
import zipfile
import tempfile
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -----------------------------
# Settings
# -----------------------------
DEFAULT_WEIGHTS = "best (1).pt"      # try this first; we also auto-discover *.pt below
PAGE_TITLE = "Casting Defects Detection"

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

# -----------------------------
# Helpers
# -----------------------------
def discover_weights() -> Path | None:
    """Return a Path to any *.pt in repo if DEFAULT_WEIGHTS missing."""
    p = Path(DEFAULT_WEIGHTS)
    if p.exists():
        return p
    cands = sorted(Path(".").glob("*.pt"))
    return cands[0] if cands else None

@st.cache_resource(show_spinner=False)
def load_model():
    w = discover_weights()
    if not w:
        st.error("No weights found. Place a .pt file in the repo root (e.g., best.pt).")
        st.stop()
    model = YOLO(str(w))
    return model, str(w.resolve())

def zip_dir(dir_path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in dir_path.rglob("*"):
            if f.is_file():
                zf.write(f, arcname=f.relative_to(dir_path))
    buf.seek(0)
    return buf.read()

# Sidebar controls
st.sidebar.header("Model")
model, resolved_path = load_model()
st.sidebar.success(f"Loaded: {Path(resolved_path).name}")

conf_thres = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
iou_thres  = st.sidebar.slider("IoU (NMS)", 0.30, 0.90, 0.50, 0.05)
imgsz      = st.sidebar.selectbox("Image size", [512, 640, 300, 320], index=1)

# -----------------------------
# Image inference UI
# -----------------------------
st.subheader("Upload image(s) for detection")
imgs = st.file_uploader(
    "Choose image files",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)

run_btn = st.button("Run detection", type="primary", disabled=not imgs)

if run_btn and imgs:
    # Use a brand-new temp folder each click; no experimental_user calls.
    run_root = Path(tempfile.mkdtemp(prefix="runs_"))
    pred_dir = run_root / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    with st.spinner("Running inference..."):
        for i, up in enumerate(imgs):
            # Save upload to disk (Ultralytics writes rendered image alongside)
            in_path = pred_dir / f"img_{i}_{Path(up.name).stem}.png"
            Image.open(up).convert("RGB").save(in_path)

            model.predict(
                source=str(in_path),
                imgsz=imgsz,
                conf=conf_thres,
                iou=iou_thres,
                save=True,
                project=str(run_root),
                name="pred",
                exist_ok=True,
                verbose=False
            )

        rendered = sorted((run_root / "pred").glob("*.jpg"))

    st.success(f"Done. Rendered {len(rendered)} image(s).")
    cols = st.columns(2 if len(rendered) > 1 else 1)
    for idx, rp in enumerate(rendered):
        with cols[idx % len(cols)]:
            st.image(str(rp), caption=rp.name, use_container_width=True)

    if rendered:
        st.download_button(
            "Download all results (ZIP)",
            data=zip_dir(run_root / "pred"),
            file_name="detections.zip",
            mime="application/zip",
        )
else:
    st.info("Upload one or more images, then click **Run detection**.")
