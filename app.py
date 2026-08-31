"""
Nigerian Food & Snacks Classifier
Streamlit App — tf_efficientnetv2_m backbone · Custom head · Grad-CAM
Built from: group33-nigerianfoodimageeclassification-notebook.ipynb
"""

import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
#  CONSTANTS — pulled directly from notebook CFG + discover_dataset
# ─────────────────────────────────────────────────────────────────
MODEL_NAME  = "tf_efficientnetv2_m"
NUM_CLASSES = 21
IMG_SIZE    = 224
CHECKPOINT  = "checkpoints/best_fold0.pth"
OOF_ACCURACY = 0.8768
OOF_MACRO_F1 = 0.7933

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Exact class order from notebook (sorted alphabetically by discover_dataset)
CLASS_NAMES = [
    "Abacha and Ugba (african salad)",  # 0
    "Akara and Eko",                    # 1
    "Akara and Eko-Akamu",              # 2
    "Amala and Ewedu-Gbegiri",          # 3
    "Amala and Gbegiri-Ewedu",          # 4
    "Asaro",                            # 5
    "Boli (bole)",                      # 6
    "Chin Chin",                        # 7
    "Egusi Soup",                       # 8
    "Ewa-Agoyin",                       # 9
    "Fried Plantains (dodo)",           # 10
    "Jollof Rice",                      # 11
    "Meat Pie",                         # 12
    "Moin-Moin",                        # 13
    "Nkwobi",                           # 14
    "Okro Soup",                        # 15
    "Pepper Soup",                      # 16
    "Pepper-Soup",                      # 17
    "Puff-Puff",                        # 18
    "Suya",                             # 19
    "Vegetable Soup",                   # 20
]

# Per-class F1 from out-of-fold validation
CLASS_F1 = {
    "Abacha and Ugba (african salad)": 0.974, "Akara and Eko": 0.012,
    "Akara and Eko-Akamu": 0.463, "Amala and Ewedu-Gbegiri": 0.295,
    "Amala and Gbegiri-Ewedu": 0.350, "Asaro": 0.987, "Boli (bole)": 0.989,
    "Chin Chin": 0.983, "Egusi Soup": 0.968, "Ewa-Agoyin": 0.986,
    "Fried Plantains (dodo)": 0.979, "Jollof Rice": 0.983, "Meat Pie": 1.000,
    "Moin-Moin": 0.969, "Nkwobi": 0.976, "Okro Soup": 0.952,
    "Pepper Soup": 0.536, "Pepper-Soup": 0.292, "Puff-Puff": 0.997,
    "Suya": 0.993, "Vegetable Soup": 0.976,
}


# ─────────────────────────────────────────────────────────────────
#  MODEL ARCHITECTURE — mirrors NigerianFoodClassifier from notebook
# ─────────────────────────────────────────────────────────────────
class NigerianFoodClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int,
                 pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool='avg',
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(p=dropout / 2),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.BatchNorm1d(feat_dim // 2),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nigerian Food Classifier",
    page_icon="🍽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  DESIGN TOKENS + CSS
#  A model card laid out like a chop-bar order ticket: paper ground,
#  hairline rules, one rust accent, tabular numerals for every figure.
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #F7F4EE;
    --surface: #FFFFFF;
    --ink: #1C1A16;
    --ink-2: #6E675C;
    --line: #E1DACB;
    --accent: #A8452F;
    --accent-ink: #7C3320;
    --accent-soft: #F2E3D9;
    --green: #45604A;
    --green-soft: #E8EEE6;
    --sans: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace;
}

html, body, [class*="css"] { font-family: var(--sans); color: var(--ink); }
.stApp { background: var(--bg); }
h1, h2, h3, h4 { font-family: var(--sans); color: var(--ink); }
p, span, label, div { font-family: var(--sans); }

/* ── Header ─────────────────────────────────────────────── */
.app-header {
    border-bottom: 1px solid var(--line);
    padding-bottom: 1.5rem;
    margin-bottom: 1.75rem;
}
.app-header h1 {
    font-size: 2.15rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 .5rem 0;
    line-height: 1.15;
}
.app-header p {
    color: var(--ink-2);
    font-size: .98rem;
    max-width: 62ch;
    line-height: 1.5;
    margin: 0 0 1.35rem 0;
}
.stat-strip {
    display: flex;
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    width: fit-content;
}
.stat-block {
    padding: .7rem 1.6rem .7rem 0;
    margin-right: 1.6rem;
    border-right: 1px solid var(--line);
}
.stat-block:last-child { border-right: none; margin-right: 0; padding-right: 0; }
.stat-block .num {
    font-family: var(--mono);
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--ink);
    display: block;
}
.stat-block .lbl {
    font-size: .78rem;
    color: var(--ink-2);
    margin-top: .1rem;
    display: block;
}

/* ── Section labels ─────────────────────────────────────── */
.section-label {
    font-size: .8rem;
    font-weight: 600;
    color: var(--ink-2);
    margin: 0 0 .6rem 0;
}

/* ── Upload zone ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 1px dashed var(--line);
    border-radius: 6px;
    padding: .25rem;
}
[data-testid="stFileUploaderDropzone"] { background: var(--surface); }

/* ── Prediction ticket ──────────────────────────────────── */
.ticket {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1.5rem 1.6rem;
}
.ticket .eyebrow {
    font-size: .78rem;
    color: var(--ink-2);
    margin: 0 0 .35rem 0;
}
.ticket .name {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin: 0 0 .9rem 0;
    line-height: 1.2;
}
.ticket .conf-row {
    display: flex;
    align-items: baseline;
    gap: .5rem;
    margin-bottom: 1rem;
}
.ticket .conf-num {
    font-family: var(--mono);
    font-size: 2.1rem;
    font-weight: 600;
    color: var(--accent-ink);
    line-height: 1;
}
.ticket .conf-lbl { color: var(--ink-2); font-size: .85rem; }
.ticket .meta-row {
    border-top: 1px solid var(--line);
    padding-top: .8rem;
    display: flex;
    gap: 1.6rem;
    font-size: .82rem;
    color: var(--ink-2);
}
.ticket .meta-row b {
    font-family: var(--mono);
    color: var(--ink);
    font-weight: 600;
}
.img-frame {
    border: 1px solid var(--line);
    border-radius: 8px;
    overflow: hidden;
    background: var(--surface);
}

/* ── Ranked list ────────────────────────────────────────── */
.rank-row {
    display: flex;
    align-items: center;
    gap: .85rem;
    padding: .55rem 0;
    border-bottom: 1px solid var(--line);
}
.rank-row:first-child { border-top: 1px solid var(--line); }
.rank-idx {
    font-family: var(--mono);
    font-size: .82rem;
    color: var(--ink-2);
    width: 1.4rem;
}
.rank-name {
    flex: 0 0 auto;
    min-width: 200px;
    max-width: 260px;
    font-size: .88rem;
    color: var(--ink);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.rank-track {
    flex: 1;
    height: 6px;
    background: var(--line);
    border-radius: 3px;
    overflow: hidden;
}
.rank-fill { height: 100%; background: var(--accent); }
.rank-fill.top { background: var(--accent-ink); }
.rank-pct {
    font-family: var(--mono);
    font-size: .85rem;
    color: var(--ink);
    width: 3.4rem;
    text-align: right;
}

/* ── Grad-CAM captions ──────────────────────────────────── */
.gcam-cap {
    font-size: .82rem;
    color: var(--ink-2);
    margin-bottom: .4rem;
}
.gcam-note {
    font-size: .85rem;
    color: var(--ink-2);
    border-top: 1px solid var(--line);
    padding-top: .7rem;
    margin-top: .9rem;
}

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-size: .95rem !important;
    font-weight: 700 !important;
    color: var(--ink) !important;
}
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] label {
    color: var(--ink-2) !important;
}
.sb-info {
    font-size: .82rem;
    color: var(--ink-2);
    line-height: 1.8;
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: .8rem .9rem;
}
.sb-info b { color: var(--ink); font-family: var(--mono); font-weight: 600; }

/* ── Buttons / progress / misc widgets ──────────────────── */
.stButton > button {
    background: var(--ink);
    color: var(--bg) !important;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: .5rem 1.3rem;
}
.stButton > button:hover { background: var(--accent-ink); }
.stProgress > div > div > div { background: var(--accent) !important; }
[data-testid="stMetricValue"] { font-family: var(--mono); color: var(--ink); }
details { background: var(--surface) !important; border: 1px solid var(--line) !important; border-radius: 6px !important; }

/* ── Empty state ────────────────────────────────────────── */
.empty-state {
    padding: 2.5rem 0 1rem 0;
    color: var(--ink-2);
}
.empty-state .lead { font-size: 1.02rem; color: var(--ink); margin-bottom: .3rem; }
.empty-state .sub { font-size: .88rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> torch.Tensor:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()


# ─────────────────────────────────────────────────────────────────
#  MODEL LOAD
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mn  = ckpt.get("model_name", MODEL_NAME)
    nc  = ckpt.get("num_classes", NUM_CLASSES)
    model = NigerianFoodClassifier(mn, nc, pretrained=False, dropout=0.3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("val_f1", None), ckpt.get("epoch", None)


# ─────────────────────────────────────────────────────────────────
#  GRAD-CAM  (hooks on backbone's last conv block)
# ─────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.grads = None
        self.acts  = None
        target = list(model.backbone.blocks)[-1][-1]
        for name in ("conv_pwl", "conv_dw", "conv_exp"):
            if hasattr(target, name):
                layer = getattr(target, name)
                break
        else:
            layer = target
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out): self.acts  = out.detach()
    def _bwd(self, _m, _i, out): self.grads = out[0].detach()

    def generate(self, tensor: torch.Tensor, cls_idx: int) -> np.ndarray:
        self.model.zero_grad()
        t = tensor.clone().requires_grad_(True)
        logits = self.model(t)
        logits[0, cls_idx].backward()
        w   = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.acts).sum(dim=1).squeeze()
        cam = F.relu(cam)
        if cam.max() > 0: cam = cam / cam.max()
        return cam.cpu().numpy()


def blend_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    import matplotlib.cm as cm
    resized  = cv2.resize(cam, (image.width, image.height))
    heatmap  = (cm.jet(resized)[:, :, :3] * 255).astype(np.uint8)
    orig     = np.array(image.convert("RGB"))
    return (alpha * heatmap + (1 - alpha) * orig).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")
    ckpt_path = st.text_input(
        "Checkpoint path", value=CHECKPOINT,
        help="Path to best_fold0.pth relative to app.py",
    )

    st.divider()
    st.markdown("### Grad-CAM")
    show_gcam  = st.toggle("Show attention map", value=True)
    cam_alpha  = st.slider("Overlay strength", 0.2, 0.8, 0.48, 0.02)
    cam_target = st.selectbox("Class to visualise", ["Top prediction"] + CLASS_NAMES)

    st.divider()
    st.markdown("### Results")
    top_k          = st.slider("Number of results shown", 3, 10, 5)
    show_all_probs = st.toggle("Show full probability table", value=False)

    st.divider()
    st.markdown("### Model")
    st.markdown(f"""
    <div class="sb-info">
    Architecture&nbsp; <b>{MODEL_NAME}</b><br>
    Classes&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>{NUM_CLASSES}</b><br>
    Input size&nbsp;&nbsp; <b>{IMG_SIZE}×{IMG_SIZE}px</b><br>
    OOF accuracy&nbsp; <b>{OOF_ACCURACY*100:.2f}%</b><br>
    Macro F1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>{OOF_MACRO_F1:.4f}</b>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("All 21 classes — validation F1"):
        class_df = pd.DataFrame({
            "#": list(range(len(CLASS_NAMES))),
            "Class": CLASS_NAMES,
            "F1": [CLASS_F1.get(c, 0) for c in CLASS_NAMES],
        })
        st.dataframe(
            class_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "#": st.column_config.NumberColumn(width="small"),
                "F1": st.column_config.ProgressColumn(
                    "F1", min_value=0, max_value=1, format="%.3f",
                ),
            },
        )


# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
  <h1>Nigerian Food Classifier</h1>
  <p>Upload a photo of a Nigerian dish or snack. The model identifies it and
  shows a Grad-CAM attention map, so you can see which part of the image
  drove its decision.</p>
  <div class="stat-strip">
    <div class="stat-block"><span class="num">{OOF_ACCURACY*100:.1f}%</span><span class="lbl">Accuracy</span></div>
    <div class="stat-block"><span class="num">{NUM_CLASSES}</span><span class="lbl">Food classes</span></div>
    <div class="stat-block"><span class="num">{IMG_SIZE}px</span><span class="lbl">Input size</span></div>
    <div class="stat-block"><span class="num">{OOF_MACRO_F1:.3f}</span><span class="lbl">Macro F1</span></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────────
model_ok = False
with st.spinner("Loading model weights…"):
    try:
        model, saved_f1, saved_ep = load_model(ckpt_path)
        model_ok = True
        detail = []
        if saved_f1: detail.append(f"F1 {saved_f1:.4f}")
        if saved_ep: detail.append(f"epoch {saved_ep}")
        detail_str = f" ({', '.join(detail)})" if detail else ""
        st.caption(f"Checkpoint loaded — `{ckpt_path}`{detail_str}")
    except FileNotFoundError:
        st.error(f"Checkpoint not found at `{ckpt_path}`. Place `best_fold0.pth` beside `app.py`.")
    except Exception as e:
        st.error(f"Could not load checkpoint: {e}")


# ─────────────────────────────────────────────────────────────────
#  FILE UPLOADER
# ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Upload image</p>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drag and drop a file, or click to browse — JPG, PNG or WEBP",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

# ─────────────────────────────────────────────────────────────────
#  INFERENCE & DISPLAY
# ─────────────────────────────────────────────────────────────────
if uploaded and model_ok:
    image  = Image.open(uploaded).convert("RGB")
    tensor = preprocess(image)

    with st.spinner("Analysing image…"):
        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).squeeze().numpy()

    top_idx   = probs.argsort()[::-1][:top_k]
    top_probs = probs[top_idx]
    top_names = [CLASS_NAMES[i] for i in top_idx]

    best_name = top_names[0]
    best_conf = top_probs[0]
    best_f1   = CLASS_F1.get(best_name, 0)
    best_idx  = int(top_idx[0])

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<p class="section-label">Uploaded image</p>', unsafe_allow_html=True)
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<p class="section-label">Prediction</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="ticket">
          <p class="eyebrow">Best match</p>
          <p class="name">{best_name}</p>
          <div class="conf-row">
            <span class="conf-num">{best_conf*100:.1f}%</span>
            <span class="conf-lbl">confidence</span>
          </div>
          <div class="meta-row">
            <span>Class <b>{best_idx:02d}</b> of {NUM_CLASSES}</span>
            <span>Validation F1 <b>{best_f1:.3f}</b></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<p class="section-label" style="margin-top:1.3rem">Top {top_k} results</p>', unsafe_allow_html=True)
        rows = ""
        for rank, (name, conf) in enumerate(zip(top_names, top_probs)):
            pct    = conf * 100
            fill   = "top" if rank == 0 else ""
            rows += f"""
            <div class="rank-row">
              <span class="rank-idx">{rank+1:02d}</span>
              <span class="rank-name">{name}</span>
              <div class="rank-track"><div class="rank-fill {fill}" style="width:{pct:.1f}%"></div></div>
              <span class="rank-pct">{pct:.1f}%</span>
            </div>"""
        st.markdown(rows, unsafe_allow_html=True)

    # ── Grad-CAM ─────────────────────────────────────────────
    if show_gcam:
        st.markdown('<p class="section-label" style="margin-top:1.8rem">Grad-CAM attention</p>', unsafe_allow_html=True)
        try:
            gcam = GradCAM(model)
            cls_idx = (CLASS_NAMES.index(cam_target)
                       if cam_target != "Top prediction"
                       else best_idx)
            cls_label = CLASS_NAMES[cls_idx]

            cam_map = gcam.generate(preprocess(image), cls_idx)
            overlay = blend_cam(image, cam_map, alpha=cam_alpha)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<p class="gcam-cap">Original</p>', unsafe_allow_html=True)
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<p class="gcam-cap">Activation map</p>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(4, 4))
                fig.patch.set_facecolor("#FFFFFF")
                ax.set_facecolor("#FFFFFF")
                resized_cam = cv2.resize(cam_map, (image.width, image.height))
                im  = ax.imshow(resized_cam, cmap="inferno")
                ax.axis("off")
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.yaxis.set_tick_params(color="#6E675C")
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#6E675C")
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight",
                            facecolor="#FFFFFF", dpi=120)
                plt.close(fig)
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(buf.getvalue(), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<p class="gcam-cap">Overlay — {cls_label}</p>', unsafe_allow_html=True)
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(overlay, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(
                f'<p class="gcam-note">Highlighted regions drove the model toward '
                f'<b style="color:var(--ink)">{cls_label}</b> '
                f'(confidence {probs[cls_idx]*100:.1f}%). Brighter regions had more influence '
                f'on the prediction.</p>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning(f"Grad-CAM could not run: {e}")

    # ── Full probability table ────────────────────────────────
    if show_all_probs:
        st.markdown('<p class="section-label" style="margin-top:1.8rem">All 21 class probabilities</p>', unsafe_allow_html=True)
        df = pd.DataFrame({
            "#": list(range(len(CLASS_NAMES))),
            "Class": CLASS_NAMES,
            "Probability": probs,
            "Validation F1": [CLASS_F1.get(c, 0) for c in CLASS_NAMES],
        }).sort_values("Probability", ascending=False).reset_index(drop=True)

        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            height=460,
            column_config={
                "#": st.column_config.NumberColumn(width="small"),
                "Probability": st.column_config.ProgressColumn(
                    "Probability", min_value=0, max_value=1, format="%.3f",
                ),
                "Validation F1": st.column_config.NumberColumn(format="%.3f"),
            },
        )

elif not model_ok:
    st.info("Set a valid checkpoint path in the sidebar, then upload an image.")

else:
    st.markdown("""
    <div class="empty-state">
      <p class="lead">No image uploaded yet.</p>
      <p class="sub">Upload a photo of a Nigerian dish or snack to get a prediction —
      Jollof Rice, Suya, Puff-Puff, Egusi Soup, Chin Chin, and 16 more.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">Supported classes</p>', unsafe_allow_html=True)
    class_df = pd.DataFrame({
        "#": list(range(len(CLASS_NAMES))),
        "Class": CLASS_NAMES,
        "Validation F1": [CLASS_F1.get(c, 0) for c in CLASS_NAMES],
    })
    st.dataframe(
        class_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Validation F1": st.column_config.ProgressColumn(
                "Validation F1", min_value=0, max_value=1, format="%.3f",
            ),
        },
    )
