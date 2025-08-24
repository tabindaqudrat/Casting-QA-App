import io, os, zipfile, tempfile, shutil, time
from pathlib import Path

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="YOLOv8 Casting Defects", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_model(model_bytes: bytes | None, model_path: str | None):
    if model_bytes:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        tmp.write(model_bytes)
        tmp.flush(); tmp.close()
        return YOLO(tmp.name), tmp.name
    else:
        if model_path and Path(model_path).exists():
            return YOLO(model_path), model_path
        st.stop()

def save_uploaded_bytes(up, suffix):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(up.read()); tmp.flush(); tmp.close()
    return tmp.name

def unzip_to_temp(zip_bytes) -> str:
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        zf.extractall(tmpdir)
    return tmpdir

def try_plot_green_cm(eval_dir: Path, title="Confusion Matrix — Test (green)"):
    """
    Look for a saved confusion matrix array from Ultralytics and replot it with a green colormap.
    Fallback: tint the existing PNG.
    """
    candidates = [
        eval_dir / "confusion_matrix.npy",
        eval_dir / "confusion_matrix_normalized.npy",
        eval_dir / "cm.npy",
    ]
    cm = None
    for p in candidates:
        if p.exists():
            try:
                cm = np.load(p)
                break
            except Exception:
                pass

    fig = plt.figure(figsize=(6,5))
    ax  = plt.gca()

    if cm is not None:
        im = ax.imshow(cm, cmap="Greens")
        # try to infer labels from shape (append background)
        n = cm.shape[0]
        # names.txt saved by Ultralytics if available
        names_txt = eval_dir / "names.txt"
        if names_txt.exists():
            names = [ln.strip() for ln in names_txt.read_text().splitlines() if ln.strip()]
        else:
            # generic labels c0..c(n-2) + background
            names = [f"c{i}" for i in range(n-1)] + ["background"]

        ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted")
        ax.set_title(title)
        # annotate
        for i in range(n):
            for j in range(n):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9)
        fig.tight_layout()
    else:
        # fallback: recolor PNG
        cm_png = eval_dir / "confusion_matrix.png"
        if not cm_png.exists():
            st.info("No confusion matrix found in eval folder.")
            return
        img = Image.open(cm_png).convert("L")
        arr = np.array(img)
        plt.imshow(arr, cmap="Greens")
        plt.axis("off")
        plt.title(title)
        fig.tight_layout()

    st.pyplot(fig, clear_figure=True)

# -----------------------------
# Sidebar: model selection
# -----------------------------
st.sidebar.header("Model")
uploaded_model = st.sidebar.file_uploader("Upload YOLOv8 .pt", type=["pt"])
default_model_path = st.sidebar.text_input("...or path to a local .pt", value="best.pt")
conf_thres = st.sidebar.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
iou_thres  = st.sidebar.slider("IoU (NMS / metrics)", 0.3, 0.9, 0.5, 0.05)
imgsz      = st.sidebar.selectbox("Image size", [512, 640, 768, 896], index=1)

with st.spinner("Loading model..."):
    model, model_used_path = load_model(uploaded_model.read() if uploaded_model else None, default_model_path)
st.sidebar.success(f"Loaded: {Path(model_used_path).name}")

# -----------------------------
# Tabs
# -----------------------------
tab_pred, tab_eval = st.tabs(["🔍 Inference", "📊 Evaluate & Green Confusion Matrix"])

# --------- Inference tab ----------
with tab_pred:
    st.subheader("Run inference on images or a short video")
    src = st.radio("Choose input", ["Image(s)", "Video"], horizontal=True)

    if src == "Image(s)":
        imgs = st.file_uploader("Upload image files", type=["png","jpg","jpeg","bmp"], accept_multiple_files=True)
        if imgs:
            cols = st.columns(2)
            out_dir = Path(tempfile.mkdtemp())
            for i, up in enumerate(imgs):
                im_path = save_uploaded_bytes(up, suffix=f"_{i}.png")
                results = model.predict(im_path, imgsz=imgsz, conf=conf_thres, iou=iou_thres, verbose=False, save=True, project=str(out_dir), name="pred")
            st.success("Predictions complete.")
            pred_dir = out_dir / "pred"
            for p in sorted(pred_dir.glob("*.jpg")):
                st.image(str(p), caption=p.name, use_container_width=True)
    else:
        vid = st.file_uploader("Upload a short MP4/MOV", type=["mp4","mov","avi","mkv"])
        if vid:
            vpath = save_uploaded_bytes(vid, suffix=".mp4")
            out_dir = Path(tempfile.mkdtemp())
            results = model.predict(vpath, imgsz=imgsz, conf=conf_thres, iou=iou_thres, verbose=False,
                                    save=True, project=str(out_dir), name="pred")
            # show first rendered frame (Streamlit can't autoplay arbitrary codecs reliably)
            rendered = sorted((out_dir/"pred").glob("*"))
            if rendered:
                st.video(str(vpath))
                st.caption("Original video (predicted file saved to temp folder).")

# --------- Evaluate tab ----------
with tab_eval:
    st.subheader("Evaluate a YOLO-format dataset and plot a GREEN confusion matrix")

    st.markdown("""
    **Option A (easiest)**: Upload a ZIP that contains your dataset folders (`images/`, `labels/`) and a `data.yaml` inside.
    \n**Option B**: Provide a path on the server to an existing `data.yaml`.
    """)

    up_zip = st.file_uploader("Upload dataset ZIP (optional)", type=["zip"])
    data_yaml_path = st.text_input("Or path to data.yaml on server (optional)", value="")
    split = st.selectbox("Split to evaluate", ["val", "test"], index=1)
    batch = st.number_input("Batch size", min_value=1, max_value=64, value=16)

    if st.button("Run Evaluation"):
        with st.spinner("Evaluating..."):
            work_dir = None
            yaml_to_use = None
            try:
                if up_zip is not None:
                    work_dir = unzip_to_temp(up_zip.read())
                    # try to locate a data.yaml in the extracted bundle
                    cands = list(Path(work_dir).rglob("data.yaml"))
                    if not cands:
                        st.error("No data.yaml found in the uploaded ZIP.")
                        st.stop()
                    yaml_to_use = str(cands[0])
                elif data_yaml_path and Path(data_yaml_path).exists():
                    yaml_to_use = data_yaml_path
                else:
                    st.error("Please upload a dataset ZIP or give a valid path to data.yaml.")
                    st.stop()

                # Run evaluation
                out_project = Path(tempfile.mkdtemp())
                metrics = model.val(
                    data=yaml_to_use,
                    split=split,
                    imgsz=imgsz,
                    batch=batch,
                    conf=conf_thres,
                    iou=iou_thres,
                    plots=True,
                    save_json=True,
                    project=str(out_project),
                    name=f"{split}_eval",
                    verbose=False
                )

                save_dir = Path(getattr(metrics, "save_dir", out_project / f"{split}_eval"))
                st.success(f"Done. Results in: {save_dir}")

                # Print key metrics
                if hasattr(metrics, "results_dict") and isinstance(metrics.results_dict, dict):
                    st.write("**Key metrics**")
                    st.json({k: float(v) for k, v in metrics.results_dict.items()})

                # Green confusion matrix
                st.write("**Green confusion matrix**")
                try_plot_green_cm(save_dir, title=f"Confusion Matrix — {split} (green)")

                # Download buttons
                cm_png = save_dir / "confusion_matrix.png"
                if cm_png.exists():
                    st.download_button("Download original confusion_matrix.png",
                                       data=cm_png.read_bytes(), file_name=cm_png.name)
                green_png = save_dir / "confusion_matrix_green.png"
                if green_png.exists():
                    st.download_button("Download green confusion matrix",
                                       data=green_png.read_bytes(), file_name=green_png.name)

                res_csv = save_dir / "results.csv"
                if res_csv.exists():
                    st.download_button("Download results.csv",
                                       data=res_csv.read_bytes(), file_name=res_csv.name)

            finally:
                # clean temporary extraction if used
                if work_dir and Path(work_dir).exists():
                    shutil.rmtree(work_dir, ignore_errors=True)
