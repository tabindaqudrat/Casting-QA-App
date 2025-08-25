# --- Minimal YOLOv8 Image Inference App (images only) ---
# Place your weights file in the repo root as 'best.pt' or change DEFAULT_WEIGHTS below.

import os
import io
import zipfile
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless backend
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -----------------------------
# Settings
# -----------------------------
DEFAULT_WEIGHTS = "best (1).pt"
   # change if your repo path is different
PAGE_TITLE = "YOLOv8 Casting Defects — Image Inference"

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    p = Path(weights_path)
    if not p.exists():
        st.error(f"Could not find weights at: `{p}`. "
                 "Make sure the file is in your repo or update DEFAULT_WEIGHTS.")
        st.stop()
    model = YOLO(str(p))
    return model, str(p.resolve())

def zip_dir(dir_path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in dir_path.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=file.relative_to(dir_path))
    buf.seek(0)
    return buf.read()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Model")
model, resolved_path = load_model(DEFAULT_WEIGHTS)
st.sidebar.success(f"Loaded: {Path(resolved_path).name}")

conf_thres = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
iou_thres  = st.sidebar.slider("IoU (NMS)", 0.30, 0.90, 0.50, 0.05)
imgsz      = st.sidebar.selectbox("Image size", [300, 320, 512, 640], index=1)

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
    # Save uploads to a temp folder (so Ultralytics can write rendered images next to them)
    out_root = Path(st.experimental_user().id if hasattr(st, "experimental_user") else "tmp_runs")
    out_dir  = out_root / "pred"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    with st.spinner("Running inference..."):
        rendered_paths: List[Path] = []
        for i, up in enumerate(imgs):
            # Save the uploaded image
            p = out_dir / f"img_{i}_{Path(up.name).stem}.png"
            Image.open(up).convert("RGB").save(p)

            # Predict and save rendered result
            model.predict(
                source=str(p),
                imgsz=imgsz,
                conf=conf_thres,
                iou=iou_thres,
                save=True,
                project=str(out_root),
                name="pred",          # ensures output is in out_root/pred
                exist_ok=True,
                verbose=False
            )

        # Collect rendered images (Ultralytics writes *.jpg by default)
        rendered_paths = sorted((out_root / "pred").glob("*.jpg"))

    st.success(f"Done. Rendered {len(rendered_paths)} image(s).")

    # Show results in a responsive grid
    cols_per_row = 2 if len(rendered_paths) > 1 else 1
    cols = st.columns(cols_per_row)
    for idx, rp in enumerate(rendered_paths):
        with cols[idx % cols_per_row]:
            st.image(str(rp), caption=rp.name, use_container_width=True)

    # Offer a ZIP download of all rendered images
    if rendered_paths:
        zip_bytes = zip_dir(out_root / "pred")
        st.download_button(
            "Download all results (ZIP)",
            data=zip_bytes,
            file_name="detections.zip",
            mime="application/zip",
        )
else:
    st.info("Upload one or more images, then click **Run detection**.")
