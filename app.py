"""
🍜 Nigerian Food & Snacks Classifier — Polished Edition
Streamlit App · tf_efficientnetv2_m backbone · Custom head · Grad-CAM

from __future__ import annotations

import io
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─────────────────────────────────────────────────────────────────
#  CONSTANTS — pulled directly from notebook CFG + discover_dataset
# ─────────────────────────────────────────────────────────────────
MODEL_NAME  = "tf_efficientnetv2_m"
NUM_CLASSES = 21
IMG_SIZE    = 224
CHECKPOINT  = "checkpoints/best_fold0.pth"

# ImageNet normalisation (standard for timm models)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Exact class order from notebook (sorted alphabetically by discover_dataset)
CLASS_NAMES = [
    "Abacha and Ugba(african salad)",   # 0
    "Akara and Eko",                    # 1
    "Akara and Eko-Akamu",              # 2
    "Amala and Ewedu-Gbegiri",          # 3
    "Amala and Gbegiri- Ewedu",         # 4
    "Asaro",                            # 5
    "Boli(bole)",                       # 6
    "Chin Chin",                        # 7
    "Egusi Soup",                       # 8
    "Ewa-Agoyin",                       # 9
    "Fried Plantains (dodo)",           # 10
    "Jollof Rice",                      # 11
    "Meat-pie",                         # 12
    "Moin-Moin",                        # 13
    "Nkwobi",                           # 14
    "Okro Soup",                        # 15
    "Pepper Soup",                      # 16
    "Pepper-Soup",                      # 17
    "Puff-Puff",                        # 18
    "Suya",                             # 19
    "Vegetable Soup",                   # 20
]

FOOD_EMOJIS = {
    "Abacha and Ugba(african salad)": "🥗",
    "Akara and Eko":                  "🫓",
    "Akara and Eko-Akamu":            "🫓",
    "Amala and Ewedu-Gbegiri":        "🍲",
    "Amala and Gbegiri- Ewedu":       "🍲",
    "Asaro":                          "🍠",
    "Boli(bole)":                     "🍌",
    "Chin Chin":                      "🍪",
    "Egusi Soup":                     "🥣",
    "Ewa-Agoyin":                     "🫘",
    "Fried Plantains (dodo)":         "🍟",
    "Jollof Rice":                    "🍚",
    "Meat-pie":                       "🥧",
    "Moin-Moin":                      "🫔",
    "Nkwobi":                         "🍖",
    "Okro Soup":                      "🥘",
    "Pepper Soup":                    "🌶️",
    "Pepper-Soup":                    "🌶️",
    "Puff-Puff":                      "🔮",
    "Suya":                           "🍢",
    "Vegetable Soup":                 "🥬",
}

# Per-class F1 from OOF (for info display)
CLASS_F1 = {
    "Abacha and Ugba(african salad)": 0.974, "Akara and Eko": 0.012,
    "Akara and Eko-Akamu": 0.463, "Amala and Ewedu-Gbegiri": 0.295,
    "Amala and Gbegiri- Ewedu": 0.350, "Asaro": 0.987, "Boli(bole)": 0.989,
    "Chin Chin": 0.983, "Egusi Soup": 0.968, "Ewa-Agoyin": 0.986,
    "Fried Plantains (dodo)": 0.979, "Jollof Rice": 0.983, "Meat-pie": 1.000,
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
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────
#  CUSTOM CSS — refined warm-AI visual system
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-deep:      #07090F;
  --bg:           #0B0F1A;
  --surface:      #11161F;
  --surface-2:    #161C28;
  --surface-3:    #1B2230;
  --border:       #232A38;
  --border-soft:  #1A2030;
  --text:         #F1F5F9;
  --text-2:       #94A3B8;
  --text-3:       #64748B;
  --gold:         #F5B841;
  --gold-soft:    #FCD9A0;
  --terracotta:   #E07856;
  --purple:       #8B5CF6;
  --green:        #10B981;
  --red:          #EF4444;
  --radius-lg:    18px;
  --radius-md:    12px;
  --radius-sm:    8px;
}

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  color: var(--text);
}

/* Streamlit background */
.stApp {
  background:
    radial-gradient(ellipse 80% 50% at 20% 0%, rgba(139,92,246,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 70% 50% at 80% 10%, rgba(245,184,65,0.05) 0%, transparent 55%),
    linear-gradient(180deg, var(--bg-deep) 0%, var(--bg) 100%);
  background-attachment: fixed;
}

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px; }

/* ════════════════════════════════════════════════════════
   HERO
   ════════════════════════════════════════════════════════ */
.hero {
  position: relative;
  background:
    radial-gradient(ellipse 60% 80% at 85% 30%, rgba(224,120,86,0.15) 0%, transparent 60%),
    radial-gradient(ellipse 50% 80% at 15% 70%, rgba(139,92,246,0.12) 0%, transparent 60%),
    linear-gradient(135deg, #0F1422 0%, #18102B 100%);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 3.2rem 3rem 2.8rem;
  margin-bottom: 2rem;
  overflow: hidden;
  box-shadow: 0 24px 60px -20px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.03);
}
.hero::after {
  content: '';
  position: absolute; left: 0; right: 0; bottom: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--gold), transparent);
  opacity: 0.5;
}
.hero-eyebrow {
  display: inline-flex; align-items: center; gap: .5rem;
  background: rgba(245,184,65,0.08);
  border: 1px solid rgba(245,184,65,0.25);
  color: var(--gold);
  font-size: .72rem; font-weight: 600; letter-spacing: .12em;
  text-transform: uppercase;
  padding: .35rem .9rem; border-radius: 999px;
  margin-bottom: 1.2rem;
}
.hero-eyebrow .dot {
  width: 6px; height: 6px; border-radius: 50%; background: var(--gold);
  box-shadow: 0 0 8px var(--gold);
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: .5; transform: scale(1.4); }
}
.hero h1 {
  font-size: 2.8rem; font-weight: 800; letter-spacing: -0.025em;
  margin: 0; line-height: 1.05;
  background: linear-gradient(135deg, #F5D87E 0%, #F5B841 50%, #E07856 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero .sub {
  color: var(--text-2); font-size: 1rem; line-height: 1.6;
  margin: 1rem auto 0; max-width: 640px;
}
.hero-stats {
  display: flex; flex-wrap: wrap; gap: 1.5rem;
  margin-top: 1.8rem; padding-top: 1.5rem;
  border-top: 1px solid var(--border-soft);
}
.hero-stat { display: flex; flex-direction: column; gap: .15rem; }
.hero-stat .v {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.4rem; font-weight: 700; color: var(--text);
}
.hero-stat .l {
  font-size: .68rem; font-weight: 600; letter-spacing: .1em;
  text-transform: uppercase; color: var(--text-3);
}
.hero-badges { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.4rem; }
.badge {
  display: inline-flex; align-items: center; gap: .35rem;
  background: rgba(139,92,246,0.08);
  border: 1px solid rgba(139,92,246,0.25);
  color: #C4B5FD;
  border-radius: 8px; font-size: .75rem; font-weight: 500;
  padding: .3rem .7rem;
}
.badge.gold {
  background: rgba(245,184,65,0.08);
  border-color: rgba(245,184,65,0.25);
  color: var(--gold-soft);
}

/* ════════════════════════════════════════════════════════
   SECTION HEADERS
   ════════════════════════════════════════════════════════ */
.section-header {
  display: flex; align-items: center; gap: .6rem;
  margin: 2rem 0 1rem;
}
.section-header .icon {
  width: 32px; height: 32px;
  display: flex; align-items: center; justify-content: center;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  font-size: 1.05rem;
}
.section-header .title {
  font-size: 1.05rem; font-weight: 700; color: var(--text);
  letter-spacing: -0.01em;
}
.section-header .sub {
  font-size: .8rem; color: var(--text-3); margin-left: auto;
  font-family: 'JetBrains Mono', monospace;
}

/* ════════════════════════════════════════════════════════
   UPLOAD ZONE
   ════════════════════════════════════════════════════════ */
[data-testid="stFileUploader"] {
  background: linear-gradient(180deg, var(--surface) 0%, var(--surface-2) 100%);
  border: 2px dashed var(--border);
  border-radius: var(--radius-lg);
  padding: 1rem;
  transition: all .25s ease;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--gold);
  background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface-3) 100%);
  box-shadow: 0 0 0 4px rgba(245,184,65,0.08);
}
[data-testid="stFileUploader"] section {
  padding: 1.8rem 1rem;
  background: transparent;
}
[data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] {
  background: transparent;
  border: none;
  min-height: 120px;
}
[data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] > div:first-child {
  display: flex; flex-direction: column; align-items: center; gap: .4rem;
}
[data-testid="stFileUploader"] svg { color: var(--gold); opacity: .9; }
[data-testid="stFileUploader"] button {
  background: var(--surface-3) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 500 !important;
  padding: .4rem 1rem !important;
  transition: all .2s ease;
}
[data-testid="stFileUploader"] button:hover {
  background: var(--gold) !important;
  color: var(--bg-deep) !important;
  border-color: var(--gold) !important;
}

/* ════════════════════════════════════════════════════════
   PREDICTION CARD
   ════════════════════════════════════════════════════════ */
.pred-card {
  position: relative;
  background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface) 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 2rem 1.5rem 1.5rem;
  overflow: hidden;
}
.pred-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--gold), var(--terracotta), var(--purple));
}
.pred-emoji-row {
  display: flex; align-items: center; justify-content: center; gap: 1rem;
  margin-bottom: .8rem;
}
.pred-emoji {
  font-size: 3.5rem; line-height: 1;
  filter: drop-shadow(0 4px 12px rgba(245,184,65,0.3));
}
.confidence-ring {
  position: relative; width: 86px; height: 86px;
}
.confidence-ring svg { transform: rotate(-90deg); }
.confidence-ring .track { stroke: var(--surface-3); }
.confidence-ring .progress {
  stroke: url(#confGrad);
  stroke-linecap: round;
  transition: stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1);
}
.confidence-ring .val {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.15rem; font-weight: 700; color: var(--text);
}
.pred-name {
  text-align: center;
  font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em;
  color: var(--text);
  margin: .5rem 0 .25rem;
}
.pred-conf-label {
  text-align: center; font-size: .78rem; color: var(--text-3);
  text-transform: uppercase; letter-spacing: .1em;
}
.pred-tags {
  display: flex; justify-content: center; gap: .5rem;
  margin-top: 1.2rem; flex-wrap: wrap;
}
.tag {
  display: inline-flex; align-items: center; gap: .35rem;
  font-size: .72rem; font-weight: 500;
  padding: .3rem .7rem; border-radius: 999px;
}
.tag-f1 {
  background: rgba(16,185,129,0.1);
  border: 1px solid rgba(16,185,129,0.3);
  color: #6EE7B7;
}
.tag-rank {
  background: rgba(245,184,65,0.1);
  border: 1px solid rgba(245,184,65,0.3);
  color: var(--gold-soft);
}

/* ════════════════════════════════════════════════════════
   PROBABILITY BARS
   ════════════════════════════════════════════════════════ */
.prob-list { display: flex; flex-direction: column; gap: .55rem; }
.prob-row {
  display: grid;
  grid-template-columns: 28px 1fr auto;
  align-items: center;
  gap: .75rem;
}
.prob-rank {
  font-family: 'JetBrains Mono', monospace;
  font-size: .75rem; font-weight: 600;
  color: var(--text-3);
  text-align: center;
}
.prob-row.top .prob-rank { color: var(--gold); }
.prob-label {
  display: flex; align-items: center; gap: .45rem;
  font-size: .85rem; color: var(--text);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.prob-label .e { font-size: 1rem; flex-shrink: 0; }
.prob-label .n {
  overflow: hidden; text-overflow: ellipsis;
}
.prob-row.top .prob-label .n { font-weight: 600; color: var(--gold-soft); }
.prob-bar-wrap {
  grid-column: 2 / 4;
  height: 8px; background: var(--surface-3);
  border-radius: 999px; overflow: hidden;
  margin-top: .25rem;
}
.prob-bar {
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--purple) 0%, var(--terracotta) 50%, var(--gold) 100%);
  transition: width 1s cubic-bezier(.4,0,.2,1);
  width: 0;
}
.prob-row.top .prob-bar {
  background: linear-gradient(90deg, var(--gold) 0%, var(--gold-soft) 100%);
  box-shadow: 0 0 12px rgba(245,184,65,0.4);
}
.prob-pct {
  font-family: 'JetBrains Mono', monospace;
  font-size: .8rem; font-weight: 600;
  color: var(--text-2);
  min-width: 55px; text-align: right;
}
.prob-row.top .prob-pct { color: var(--gold); }

/* ════════════════════════════════════════════════════════
   IMAGE FRAMES
   ════════════════════════════════════════════════════════ */
.img-frame {
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: .55rem;
  background: var(--surface);
}
.img-caption {
  display: flex; align-items: center; justify-content: space-between;
  margin-top: .55rem; padding: 0 .3rem;
  font-size: .75rem; color: var(--text-3);
}
.img-caption .label {
  display: flex; align-items: center; gap: .35rem;
  font-weight: 600; color: var(--text-2);
}

/* ════════════════════════════════════════════════════════
   GRAD-CAM TRIPTYCH
   ════════════════════════════════════════════════════════ */
.cam-legend {
  display: flex; align-items: center; gap: .8rem;
  margin-top: 1rem; padding: .8rem 1rem;
  background: var(--surface);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-sm);
  font-size: .78rem; color: var(--text-2);
}
.cam-legend .gradient-bar {
  flex: 1; height: 8px; border-radius: 999px;
  background: linear-gradient(90deg, #1E3A8A 0%, #06B6D4 25%, #10B981 50%, #FACC15 75%, #DC2626 100%);
}
.cam-legend .end { font-family: 'JetBrains Mono', monospace; font-size: .7rem; }

/* ════════════════════════════════════════════════════════
   EMPTY STATE
   ════════════════════════════════════════════════════════ */
.empty-state {
  text-align: center;
  padding: 3.5rem 1.5rem 2.5rem;
  background: var(--surface);
  border: 1px dashed var(--border);
  border-radius: var(--radius-lg);
  margin: 1rem 0 2rem;
}
.empty-state .icon {
  font-size: 3.5rem; margin-bottom: 1rem;
  filter: grayscale(0.2) opacity(0.85);
}
.empty-state .title {
  font-size: 1.25rem; font-weight: 700; color: var(--text);
  margin-bottom: .5rem;
}
.empty-state .desc {
  font-size: .9rem; color: var(--text-3); line-height: 1.6;
  max-width: 480px; margin: 0 auto;
}
.empty-state .examples {
  display: flex; flex-wrap: wrap; justify-content: center; gap: .5rem;
  margin-top: 1.5rem;
}
.example-chip {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: .35rem .85rem;
  font-size: .78rem; color: var(--text-2);
  display: inline-flex; align-items: center; gap: .35rem;
}

/* ════════════════════════════════════════════════════════
   CLASS GRID
   ════════════════════════════════════════════════════════ */
.class-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: .8rem;
  margin-top: 1rem;
}
.class-card {
  background: var(--surface);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-md);
  padding: .9rem;
  transition: all .2s ease;
  display: flex; flex-direction: column; gap: .35rem;
}
.class-card:hover {
  border-color: var(--gold);
  background: var(--surface-2);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px -8px rgba(0,0,0,0.5);
}
.class-card .head {
  display: flex; align-items: center; gap: .5rem;
}
.class-card .head .e { font-size: 1.4rem; }
.class-card .head .n {
  font-size: .82rem; font-weight: 600; color: var(--text);
  line-height: 1.25; flex: 1;
}
.class-card .f1-row {
  display: flex; align-items: center; justify-content: space-between;
  font-size: .7rem; color: var(--text-3);
  font-family: 'JetBrains Mono', monospace;
}
.class-card .f1-bar {
  height: 4px; background: var(--surface-3); border-radius: 999px; overflow: hidden;
}
.class-card .f1-fill {
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--terracotta), var(--gold));
}

/* ════════════════════════════════════════════════════════
   SIDEBAR
   ════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #060911 0%, #0B0F1A 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"]::before {
  content: '';
  position: absolute; top: 0; right: 0; bottom: 0; width: 1px;
  background: linear-gradient(180deg, transparent, var(--gold), transparent);
  opacity: 0.3;
}
[data-testid="stSidebar"] .stMarkdown { color: var(--text-2); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  color: var(--text) !important;
  font-size: .95rem !important; font-weight: 700 !important;
  letter-spacing: -0.01em;
}
.sidebar-section {
  padding: .8rem 0;
  border-bottom: 1px solid var(--border-soft);
}
.sidebar-section:last-child { border-bottom: none; }
.sidebar-section .label {
  font-size: .68rem; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; color: var(--text-3);
  margin-bottom: .6rem;
  display: flex; align-items: center; gap: .4rem;
}
.sidebar-meta {
  display: grid; grid-template-columns: 1fr 1fr; gap: .5rem;
  margin-top: .8rem;
}
.sidebar-meta-item {
  background: var(--surface);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-sm);
  padding: .55rem .7rem;
}
.sidebar-meta-item .v {
  font-family: 'JetBrains Mono', monospace;
  font-size: .9rem; font-weight: 700; color: var(--gold);
}
.sidebar-meta-item .l {
  font-size: .65rem; color: var(--text-3);
  text-transform: uppercase; letter-spacing: .08em;
  margin-top: .15rem;
}

/* ════════════════════════════════════════════════════════
   STREAMLIT NATIVE OVERRIDES
   ════════════════════════════════════════════════════════ */
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--purple), var(--gold)) !important;
}

.stButton > button {
  background: linear-gradient(135deg, #4C1D95 0%, #1E3A8A 100%) !important;
  color: var(--gold-soft) !important;
  border: 1px solid rgba(245,184,65,0.3) !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  padding: .55rem 1.5rem !important;
  width: 100%;
  transition: all .2s ease;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #5B21B6 0%, #1E40AF 100%) !important;
  box-shadow: 0 4px 16px -4px rgba(139,92,246,0.4);
  transform: translateY(-1px);
}

/* Toggle */
[data-testid="stToggle"] [role="switch"] {
  background: var(--surface-3);
}
[data-testid="stToggle"] [aria-checked="true"] [data-testid="stToggleSwitchKnob"] {
  background: var(--gold);
}

/* Sliders */
[data-testid="stSlider"] [data-baseweb="slider"] {
  margin: 0;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--gold);
  border-color: var(--bg-deep);
}
[data-testid="stSlider"] [data-baseweb="slider"] [class*="track-"] {
  background: var(--surface-3);
}
[data-testid="stSlider"] [data-baseweb="slider"] [class*="track-fill-"] {
  background: var(--gold);
}

/* Text input */
[data-testid="stTextInput"] input {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'JetBrains Mono', monospace;
  font-size: .82rem;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px rgba(245,184,65,0.1) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  border-radius: var(--radius-sm) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div {
  color: var(--text) !important;
  font-size: .85rem;
}

/* Expander / details */
details {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
}
details summary {
  color: var(--text) !important;
  font-weight: 600 !important;
}

/* Status messages */
.stAlert { border-radius: var(--radius-md) !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
  border-radius: var(--radius-md);
  overflow: hidden;
}
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
#  GRAD-CAM
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
    resized  = cv2.resize(cam, (image.width, image.height))
    heatmap  = (cm.jet(resized)[:, :, :3] * 255).astype(np.uint8)
    orig     = np.array(image.convert("RGB"))
    return (alpha * heatmap + (1 - alpha) * orig).astype(np.uint8)


def conf_color(pct: float) -> str:
    """Return color tag based on confidence level."""
    if pct >= 70:
        return ("High confidence", "#6EE7B7")
    if pct >= 40:
        return ("Moderate confidence", "#FCD9A0")
    return ("Low confidence", "#FCA5A5")


# ─────────────────────────────────────────────────────────────────
#  SVG CONFIDENCE RING HELPER
# ─────────────────────────────────────────────────────────────────
def confidence_ring_svg(pct: float, size: int = 86) -> str:
    r = (size - 10) / 2
    c = 2 * 3.14159 * r
    offset = c * (1 - pct / 100)
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <defs>
        <linearGradient id="confGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#F5B841"/>
          <stop offset="100%" stop-color="#E07856"/>
        </linearGradient>
      </defs>
      <circle class="track" cx="{size/2}" cy="{size/2}" r="{r}"
              stroke-width="6" fill="none"/>
      <circle class="progress" cx="{size/2}" cy="{size/2}" r="{r}"
              stroke-width="6" fill="none"
              stroke-dasharray="{c:.2f}"
              stroke-dashoffset="{offset:.2f}"/>
    </svg>
    """


# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.caption("Fine-tune model loading and visualization options")

    with st.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="label">📁 Model checkpoint</div>', unsafe_allow_html=True)
        ckpt_path = st.text_input(
            "Checkpoint path", value=CHECKPOINT,
            help="Path to best_fold0.pth relative to app.py",
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="label">🔥 Grad-CAM</div>', unsafe_allow_html=True)
        show_gcam  = st.toggle("Enable Grad-CAM", value=True)
        cam_alpha  = st.slider("Heatmap blend", 0.2, 0.8, 0.48, 0.02)
        cam_target = st.selectbox("Visualise class", ["Top prediction"] + CLASS_NAMES)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="label">📊 Results</div>', unsafe_allow_html=True)
        top_k          = st.slider("Top-K predictions", 3, 10, 5)
        show_all_probs = st.toggle("Show full probability table", value=False)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="label">🧠 Model specs</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sidebar-meta">
          <div class="sidebar-meta-item"><div class="v">EffNetV2-M</div><div class="l">Backbone</div></div>
          <div class="sidebar-meta-item"><div class="v">21</div><div class="l">Classes</div></div>
          <div class="sidebar-meta-item"><div class="v">224px</div><div class="l">Input</div></div>
          <div class="sidebar-meta-item"><div class="v">87.7%</div><div class="l">Accuracy</div></div>
          <div class="sidebar-meta-item"><div class="v">0.793</div><div class="l">Macro F1</div></div>
          <div class="sidebar-meta-item"><div class="v">PyTorch</div><div class="l">Framework</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("🏷️ All 21 classes (with OOF F1)"):
        for name in CLASS_NAMES:
            f1 = CLASS_F1.get(name, 0)
            st.markdown(
                f"{FOOD_EMOJIS.get(name,'🍽️')} **{name}**  \n"
                f"<small style='color:#64748B;font-family:JetBrains Mono,monospace'>"
                f"F1: {f1:.3f}</small>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow"><span class="dot"></span> Deep Learning · Computer Vision</div>
  <h1>Nigerian Food Classifier</h1>
  <p class="sub">
    Upload a photo of any Nigerian dish or snack — the AI identifies it instantly
    with Grad-CAM explainability so you can see <em>why</em> it decided.
  </p>
  <div class="hero-badges">
    <span class="badge gold">⏱ Real-time inference</span>
    <span class="badge">EfficientNetV2-M</span>
    <span class="badge">21 Food Classes</span>
    <span class="badge">Grad-CAM</span>
    <span class="badge">PyTorch + timm</span>
  </div>
  <div class="hero-stats">
    <div class="hero-stat"><div class="v">87.7%</div><div class="l">OOF Accuracy</div></div>
    <div class="hero-stat"><div class="v">0.793</div><div class="l">Macro F1</div></div>
    <div class="hero-stat"><div class="v">21</div><div class="l">Food Classes</div></div>
    <div class="hero-stat"><div class="v">~50ms</div><div class="l">Inference Time</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────────
model_ok = False
model_load_status = st.empty()
with st.spinner("⏳ Loading model weights…"):
    try:
        model, saved_f1, saved_ep = load_model(ckpt_path)
        model_ok = True
        f1_str = f" · F1 {saved_f1:.4f}" if saved_f1 else ""
        ep_str = f" · Epoch {saved_ep}" if saved_ep else ""
        model_load_status.success(f"✅ Model loaded from `{ckpt_path}`{f1_str}{ep_str}")
    except FileNotFoundError:
        model_load_status.error(f"❌ `{ckpt_path}` not found. Place `best_fold0.pth` beside `app.py`.")
    except Exception as e:
        model_load_status.error(f"❌ Load error: {e}")


# ─────────────────────────────────────────────────────────────────
#  UPLOAD
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
  <div class="icon">📤</div>
  <div class="title">Upload an Image</div>
  <div class="sub">JPG · PNG · WEBP</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drag & drop or click to browse — JPG / PNG / WEBP",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)


# ─────────────────────────────────────────────────────────────────
#  INFERENCE & DISPLAY
# ─────────────────────────────────────────────────────────────────
if uploaded and model_ok:
    image  = Image.open(uploaded).convert("RGB")
    tensor = preprocess(image)

    with st.spinner("🔍 Analysing image…"):
        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).squeeze().numpy()

    top_idx   = probs.argsort()[::-1][:top_k]
    top_probs = probs[top_idx]
    top_names = [CLASS_NAMES[i] for i in top_idx]

    best_name  = top_names[0]
    best_conf  = top_probs[0]
    best_emoji = FOOD_EMOJIS.get(best_name, "🍽️")
    best_f1    = CLASS_F1.get(best_name, 0)
    best_pct   = best_conf * 100
    conf_label, conf_clr = conf_color(best_pct)

    # ── Two-column layout ────────────────────────────────────
    col_l, col_r = st.columns([1, 1.05], gap="large")

    with col_l:
        st.markdown("""
        <div class="section-header">
          <div class="icon">🖼️</div>
          <div class="title">Uploaded Image</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="section-header">
          <div class="icon">🎯</div>
          <div class="title">Top Prediction</div>
          <div class="sub">MODEL OUTPUT</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="pred-card">
          <div class="pred-emoji-row">
            <div class="pred-emoji">{best_emoji}</div>
            <div class="confidence-ring">
              {confidence_ring_svg(best_pct)}
              <div class="val">{best_pct:.0f}%</div>
            </div>
          </div>
          <div class="pred-name">{best_name}</div>
          <div class="pred-conf-label" style="color:{conf_clr}">{conf_label}</div>
          <div class="pred-tags">
            <span class="tag tag-rank">#1 Prediction</span>
            <span class="tag tag-f1">OOF F1: {best_f1:.3f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="section-header" style="margin-top:1.8rem">
          <div class="icon">📊</div>
          <div class="title">Top {top_k} Predictions</div>
          <div class="sub">PROBABILITY</div>
        </div>
        """, unsafe_allow_html=True)

        # Wrap the list in its own container, then render each row via its own
        # st.markdown call. A single concatenated HTML string would be too long
        # and Streamlit would fall back to showing raw markup as code.
        st.markdown('<div class="prob-list">', unsafe_allow_html=True)
        for rank, (name, conf) in enumerate(zip(top_names, top_probs)):
            emoji = FOOD_EMOJIS.get(name, "🍽️")
            pct   = conf * 100
            is_top = rank == 0
            top_cls = " top" if is_top else ""
            st.markdown(f"""
            <div class="prob-row{top_cls}">
              <div class="prob-rank">#{rank+1}</div>
              <div class="prob-label">
                <span class="e">{emoji}</span>
                <span class="n">{name}</span>
              </div>
              <div class="prob-pct">{pct:.1f}%</div>
              <div class="prob-bar-wrap">
                <div class="prob-bar" style="width:{pct:.1f}%"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Grad-CAM ─────────────────────────────────────────────
    if show_gcam:
        st.markdown("""
        <div class="section-header" style="margin-top:2.5rem">
          <div class="icon">🔥</div>
          <div class="title">Grad-CAM Explainability</div>
          <div class="sub">WHY THE MODEL DECIDED</div>
        </div>
        """, unsafe_allow_html=True)

        try:
            gcam = GradCAM(model)
            cls_idx = (CLASS_NAMES.index(cam_target)
                       if cam_target != "Top prediction"
                       else int(top_idx[0]))
            cls_label = CLASS_NAMES[cls_idx]

            cam_map = gcam.generate(preprocess(image), cls_idx)
            overlay = blend_cam(image, cam_map, alpha=cam_alpha)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                st.markdown("""
                <div class="img-caption">
                  <div class="label">🖼️ Original</div>
                  <div>Input image</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(4, 4))
                fig.patch.set_facecolor("#11161F")
                ax.set_facecolor("#11161F")
                resized_cam = cv2.resize(cam_map, (image.width, image.height))
                im = ax.imshow(resized_cam, cmap="jet")
                ax.axis("off")
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.yaxis.set_tick_params(color="#94A3B8")
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#94A3B8")
                cbar.outline.set_edgecolor("#232A38")
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight",
                            facecolor="#11161F", dpi=120)
                plt.close(fig)
                st.image(buf.getvalue(), use_container_width=True)
                st.markdown("""
                <div class="img-caption">
                  <div class="label">🌡️ Activation Map</div>
                  <div>Raw Grad-CAM</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(overlay, use_container_width=True)
                st.markdown(f"""
                <div class="img-caption">
                  <div class="label">🎯 Overlay</div>
                  <div>{cls_label}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="cam-legend">
              <span class="end">Low</span>
              <div class="gradient-bar"></div>
              <span class="end">High</span>
              <span style="margin-left:1rem;color:var(--text-3);">·</span>
              <span>Highlighted regions drove the model toward
                <b style="color:var(--gold)">{cls_label}</b>
                (confidence <b style="color:var(--gold)">{probs[cls_idx]*100:.1f}%</b>)
              </span>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"⚠️ Grad-CAM could not run: {e}")

    # ── Full probability table ────────────────────────────────
    if show_all_probs:
        st.markdown("""
        <div class="section-header" style="margin-top:2.5rem">
          <div class="icon">📋</div>
          <div class="title">All 21 Class Probabilities</div>
          <div class="sub">FULL DISTRIBUTION</div>
        </div>
        """, unsafe_allow_html=True)
        import pandas as pd
        df = pd.DataFrame({
            "Class": [f"{FOOD_EMOJIS.get(c,'🍽️')}  {c}" for c in CLASS_NAMES],
            "Probability (%)": np.round(probs * 100, 3),
            "OOF F1": [CLASS_F1.get(c, 0) for c in CLASS_NAMES],
        }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        st.dataframe(df, use_container_width=True, height=460)

elif not model_ok:
    st.markdown("""
    <div class="empty-state">
      <div class="icon">⚙️</div>
      <div class="title">Model not loaded</div>
      <div class="desc">Fix the checkpoint path in the sidebar to enable predictions.
        Once the model loads, upload an image and the classifier will identify
        the Nigerian dish in real time.</div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Empty state with examples ────────────────────────────
    st.markdown("""
    <div class="empty-state">
      <div class="icon">📸</div>
      <div class="title">Ready to classify</div>
      <div class="desc">Upload a photo of any Nigerian dish or snack — Jollof Rice,
        Suya, Puff-Puff, Egusi Soup, Chin Chin and 16 more — and get an instant
        prediction with Grad-CAM explainability.</div>
      <div class="examples">
        <span class="example-chip">🍚 Jollof Rice</span>
        <span class="example-chip">🍢 Suya</span>
        <span class="example-chip">🔮 Puff-Puff</span>
        <span class="example-chip">🥣 Egusi Soup</span>
        <span class="example-chip">🍪 Chin Chin</span>
        <span class="example-chip">🥧 Meat-Pie</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Class grid browser ──────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="icon">🏷️</div>
      <div class="title">All Supported Classes</div>
      <div class="sub">21 NIGERIAN DISHES</div>
    </div>
    """, unsafe_allow_html=True)

    # Streamlit-native grid (renders each card via its own st.markdown call,
    # avoiding the raw-HTML-as-code fallback that happens with one giant block).
    COLS = 3
    for row_start in range(0, len(CLASS_NAMES), COLS):
        row_classes = CLASS_NAMES[row_start:row_start + COLS]
        cols = st.columns(COLS)
        for col, name in zip(cols, row_classes):
            f1 = CLASS_F1.get(name, 0)
            emoji = FOOD_EMOJIS.get(name, "🍽️")
            f1_pct = int(f1 * 100)
            with col:
                st.markdown(f"""
                <div class="class-card">
                  <div class="head">
                    <span class="e">{emoji}</span>
                    <span class="n">{name}</span>
                  </div>
                  <div class="f1-row">
                    <span>OOF F1</span>
                    <span>{f1:.3f}</span>
                  </div>
                  <div class="f1-bar"><div class="f1-fill" style="width:{f1_pct}%"></div></div>
                </div>
                """, unsafe_allow_html=True)
