"""
app.py  —  PhytoScan AI  |  Plant Disease Detection
=====================================================
UI   : PhytoScan AI (dark sci-fi, Three.js hero, glass cards, neon, animated arc)
Logic: Original app.py pipeline — unchanged
         · 6-signal leaf_validator (validate_leaf_image / get_validation_debug_info)
         · MobileNetV2 16-output model (.keras)
         · Class list from plant_classes.json (label_map, indices 0-10 mapped, 11-15 → Unknown)
         · Confidence threshold 0.75
         · Progress bar stages (Stage 1/3 → 2/3 → 3/3)
         · Debug sidebar checkbox

Supported crops: Tomato · Potato · Bell Pepper ONLY
"""

import json
import os
import time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import tensorflow as tf

from leaf_validator import validate_leaf_image, get_validation_debug_info

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PhytoScan AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (identical to app.py)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH           = "plant_disease_model.keras"
CLASSES_PATH         = "plant_classes.json"
IMG_SIZE             = (224, 224)
CONFIDENCE_THRESHOLD = 0.75

# ─────────────────────────────────────────────────────────────────────────────
# Load class names from JSON
# ─────────────────────────────────────────────────────────────────────────────

with open(CLASSES_PATH) as f:
    _classes_data = json.load(f)

_label_map    = _classes_data["label_map"]
_index_to_cls = {v: k for k, v in _label_map.items()}
NUM_MODEL_OUTPUTS = 16

ALL_CLASS_NAMES = [
    _index_to_cls.get(i, f"Unknown_class_{i}")
    for i in range(NUM_MODEL_OUTPUTS)
]

# ─────────────────────────────────────────────────────────────────────────────
# Disease info database
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_INFO = {
    "pepper bell bacterial spot": {
        "crop": "Bell Pepper", "common_name": "Bacterial Spot", "severity": "Moderate",
        "treatment": "Apply copper-based bactericides. Avoid overhead irrigation. "
                     "Remove and destroy infected plant debris.",
    },
    "pepper bell healthy": {
        "crop": "Bell Pepper", "common_name": "Healthy", "severity": "None",
        "treatment": "No action required. Continue regular monitoring and watering.",
    },
    "potato early blight": {
        "crop": "Potato", "common_name": "Early Blight", "severity": "Moderate",
        "treatment": "Apply chlorothalonil or mancozeb fungicide. "
                     "Remove infected leaves. Ensure adequate spacing for airflow.",
    },
    "potato late blight": {
        "crop": "Potato", "common_name": "Late Blight", "severity": "Severe",
        "treatment": "Apply systemic fungicide (metalaxyl) immediately. "
                     "Destroy all infected plant material. Avoid watering in the evening.",
    },
    "potato healthy": {
        "crop": "Potato", "common_name": "Healthy", "severity": "None",
        "treatment": "No action required. Continue regular monitoring and watering.",
    },
    "tomato bacterial spot": {
        "crop": "Tomato", "common_name": "Bacterial Spot", "severity": "Moderate",
        "treatment": "Apply copper-based bactericide. Avoid wetting foliage. Use disease-free seeds.",
    },
    "tomato early blight": {
        "crop": "Tomato", "common_name": "Early Blight", "severity": "Moderate",
        "treatment": "Apply azoxystrobin or chlorothalonil. "
                     "Remove lower infected leaves. Mulch around the base.",
    },
    "tomato late blight": {
        "crop": "Tomato", "common_name": "Late Blight", "severity": "Severe",
        "treatment": "Apply fungicide immediately. Destroy infected plants. "
                     "Avoid overhead irrigation and crowded planting.",
    },
    "tomato leaf mold": {
        "crop": "Tomato", "common_name": "Leaf Mold", "severity": "Moderate",
        "treatment": "Improve greenhouse ventilation. Apply mancozeb or copper fungicide. Reduce humidity.",
    },
    "tomato septoria leaf spot": {
        "crop": "Tomato", "common_name": "Septoria Leaf Spot", "severity": "Moderate",
        "treatment": "Apply fungicide (chlorothalonil). Remove infected leaves. "
                     "Avoid working with wet plants.",
    },
    "tomato target spot": {
        "crop": "Tomato", "common_name": "Target Spot", "severity": "Moderate",
        "treatment": "Apply fungicide (azoxystrobin). Remove infected leaves. "
                     "Ensure good air circulation.",
    },
    "tomato tomato mosaic virus": {
        "crop": "Tomato", "common_name": "Mosaic Virus", "severity": "Severe",
        "treatment": "No cure available. Remove infected plants. Disinfect tools. "
                     "Control aphid vectors. Use resistant varieties.",
    },
    "tomato healthy": {
        "crop": "Tomato", "common_name": "Healthy", "severity": "None",
        "treatment": "No action required. Continue regular monitoring and watering.",
    },
}

SEVERITY_COLOR = {"None": "#00ff80", "Moderate": "#f59e0b", "Severe": "#ff4d6d"}

# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def format_label(label: str) -> str:
    return label.replace("_", " ").title()

# ─────────────────────────────────────────────────────────────────────────────
# PhytoScan UI result renderer  (replaces render_result from app.py)
# ─────────────────────────────────────────────────────────────────────────────

def render_result(result: dict):
    status = result["status"]

    # ── INVALID LEAF ──────────────────────────────────────────────────────────
    if status == "invalid_leaf":
        st.markdown(f"""
        <div class="phyto-section glass" style="border-color:#f59e0b44;
             box-shadow:0 0 60px rgba(245,158,11,.2),0 4px 60px rgba(0,0,0,.5);
             text-align:center;padding:2.5rem;">
          <div style="font-size:3rem;margin-bottom:1rem;">🚫</div>
          <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.5rem;
                      color:#f59e0b;letter-spacing:.06em;margin-bottom:.8rem;">
            INVALID IMAGE — NOT A PLANT LEAF
          </div>
          <div class="neon-divider" style="background:linear-gradient(90deg,transparent,rgba(245,158,11,.4),transparent);"></div>
          <p style="font-family:'Share Tech Mono',monospace;font-size:.78rem;
                    color:rgba(245,200,80,.7);letter-spacing:.08em;line-height:1.8;margin-top:1rem;">
            {result["detail"]}<br><br>
            ⚠️ This app <strong style="color:#f59e0b;">only accepts</strong> clear,
            close-up leaf photos of:<br>
            <strong style="color:#f59e0b;">🌶 Bell Pepper &nbsp;|&nbsp; 🥔 Potato &nbsp;|&nbsp; 🍅 Tomato</strong>
          </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("💡 Tips for a good image"):
            st.markdown("""
            <div style="font-family:'Share Tech Mono',monospace;font-size:.78rem;
                        color:rgba(0,255,120,.7);line-height:2;letter-spacing:.05em;">
            • Photograph a single leaf close-up<br>
            • Use natural daylight<br>
            • Keep the leaf in focus and fill the frame<br>
            • No screenshots, diagrams, or collages<br>
            • Avoid selfies, sky, or landscape photos
            </div>
            """, unsafe_allow_html=True)
        return

    # ── LOW CONFIDENCE ────────────────────────────────────────────────────────
    if status == "low_confidence":
        st.markdown(f"""
        <div class="phyto-section glass" style="border-color:#f59e0b44;
             box-shadow:0 0 40px rgba(245,158,11,.15),0 4px 40px rgba(0,0,0,.5);
             text-align:center;padding:2rem;">
          <div style="font-size:2.5rem;margin-bottom:.8rem;">⚠️</div>
          <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.3rem;
                      color:#f59e0b;letter-spacing:.06em;margin-bottom:.6rem;">
            LOW CONFIDENCE — UNCLEAR IMAGE
          </div>
          <div class="neon-divider" style="background:linear-gradient(90deg,transparent,rgba(245,158,11,.4),transparent);"></div>
          <p style="font-family:'Share Tech Mono',monospace;font-size:.75rem;
                    color:rgba(245,200,80,.65);letter-spacing:.06em;line-height:1.8;margin-top:.8rem;">
            {result["detail"]}
          </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("💡 Tips for better accuracy"):
            st.markdown("""
            <div style="font-family:'Share Tech Mono',monospace;font-size:.78rem;
                        color:rgba(0,255,120,.7);line-height:2;letter-spacing:.05em;">
            • Use a higher-resolution photo<br>
            • Photograph a single leaf against a plain background<br>
            • Make sure the affected area is clearly visible
            </div>
            """, unsafe_allow_html=True)
        return

    # ── UNKNOWN CLASS ─────────────────────────────────────────────────────────
    if status == "unknown_class":
        st.markdown(f"""
        <div class="phyto-section glass" style="border-color:#7c3aed44;
             box-shadow:0 0 40px rgba(124,58,237,.15),0 4px 40px rgba(0,0,0,.5);
             text-align:center;padding:2rem;">
          <div style="font-size:2.5rem;margin-bottom:.8rem;">🔮</div>
          <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.3rem;
                      color:#a855f7;letter-spacing:.06em;margin-bottom:.6rem;">
            UNABLE TO IDENTIFY DISEASE
          </div>
          <div class="neon-divider" style="background:linear-gradient(90deg,transparent,rgba(168,85,247,.4),transparent);"></div>
          <p style="font-family:'Share Tech Mono',monospace;font-size:.75rem;
                    color:rgba(200,150,255,.65);letter-spacing:.06em;line-height:1.8;margin-top:.8rem;">
            {result["detail"]}
          </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── VALID RESULT ──────────────────────────────────────────────────────────
    info       = result["info"]
    conf_pct   = result["confidence"] * 100
    severity   = info.get("severity", "Unknown")
    sev_col    = SEVERITY_COLOR.get(severity, "#607D8B")
    is_healthy = severity == "None"
    crop       = info.get("crop", "—")
    disease    = info.get("common_name", "—")
    treatment  = info.get("treatment", "Consult a local agricultural extension service.")
    raw_preds  = result["raw_preds"]

    status_color = "#00ff80" if is_healthy else "#ff4d6d"
    glow_color   = "rgba(0,255,120,.3)" if is_healthy else "rgba(255,77,109,.3)"
    badge_icon   = "✅" if is_healthy else "🦠"

    # Main result card
    st.markdown(f"""
    <div class="phyto-section glass" style="border-color:{status_color}44;
         box-shadow:0 0 60px {glow_color},0 4px 60px rgba(0,0,0,.5);">

      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
        <div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:.68rem;
                      letter-spacing:.2em;color:rgba(150,200,170,.5);margin-bottom:.5rem;">
            DIAGNOSIS RESULT
          </div>
          <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.6rem;
                      color:{status_color};letter-spacing:.05em;line-height:1.1;">
            {badge_icon} {crop} — {disease}
          </div>
        </div>
        <div style="text-align:right;">
          <div class="big-num">{conf_pct:.1f}<span style="font-size:1.4rem;opacity:.6;">%</span></div>
          <div class="stat-label">Confidence</div>
        </div>
      </div>

      <div class="neon-divider"></div>

      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.2rem;">
        <div style="text-align:center;">
          <div style="font-family:'Share Tech Mono',monospace;font-size:.62rem;
                      letter-spacing:.18em;color:rgba(150,200,170,.45);margin-bottom:.4rem;">CROP</div>
          <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.1rem;
                      color:#00cfff;">{crop}</div>
        </div>
        <div style="text-align:center;">
          <div style="font-family:'Share Tech Mono',monospace;font-size:.62rem;
                      letter-spacing:.18em;color:rgba(150,200,170,.45);margin-bottom:.4rem;">DISEASE</div>
          <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.1rem;
                      color:{status_color};">{disease}</div>
        </div>
        <div style="text-align:center;">
          <div style="font-family:'Share Tech Mono',monospace;font-size:.62rem;
                      letter-spacing:.18em;color:rgba(150,200,170,.45);margin-bottom:.4rem;">SEVERITY</div>
          <div style="padding:.2rem .9rem;border-radius:999px;display:inline-block;
                      font-family:'Share Tech Mono',monospace;font-size:.72rem;letter-spacing:.12em;
                      color:{sev_col};border:1px solid {sev_col}88;background:{sev_col}14;">
            {severity.upper()}
          </div>
        </div>
      </div>

      <div class="neon-divider"></div>

      <div style="font-family:'Share Tech Mono',monospace;font-size:.68rem;
                  letter-spacing:.18em;color:rgba(150,200,170,.5);margin-bottom:.7rem;">
        🌿 RECOMMENDED TREATMENT
      </div>
      <p style="font-family:'Exo 2',sans-serif;font-size:.85rem;
                color:rgba(200,230,215,.85);line-height:1.75;letter-spacing:.02em;">
        {treatment}
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Animated confidence arc (Three.js)
    arc_color = "0x00ff80" if is_healthy else "0xff4d6d"
    arc_hex   = "#00ff80" if is_healthy else "#ff4d6d"
    components.html(f"""
    <!DOCTYPE html><html><head>
    <style>
      *{{margin:0;padding:0;box-sizing:border-box;}}
      body{{background:transparent;overflow:hidden;display:flex;
            align-items:center;justify-content:center;}}
      canvas{{display:block;}}
      .center{{position:absolute;text-align:center;pointer-events:none;}}
      .pct{{font-family:'Exo 2',sans-serif;font-weight:900;font-size:1.8rem;
            color:{arc_hex};filter:drop-shadow(0 0 12px {arc_hex});}}
      .lbl{{font-family:'Share Tech Mono',monospace;font-size:.55rem;
            letter-spacing:.2em;color:rgba(150,200,170,.5);}}
      @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@900&family=Share+Tech+Mono&display=swap');
    </style></head>
    <body>
    <canvas id="c"></canvas>
    <div class="center">
      <div class="pct">{conf_pct:.0f}%</div>
      <div class="lbl">CONFIDENCE</div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
      const W=300,H=160;
      const renderer=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),alpha:true,antialias:true}});
      renderer.setSize(W,H);
      const scene=new THREE.Scene();
      const camera=new THREE.PerspectiveCamera(50,W/H,.1,100);
      camera.position.z=7;
      const conf={conf_pct/100};
      const fullGeo=new THREE.TorusGeometry(2.5,.12,8,100,Math.PI*2);
      const fullMat=new THREE.MeshBasicMaterial({{color:0x1a2a1a,transparent:true,opacity:.3}});
      scene.add(new THREE.Mesh(fullGeo,fullMat));
      const arcGeo=new THREE.TorusGeometry(2.5,.18,8,100,Math.PI*2*conf);
      const arcMat=new THREE.MeshBasicMaterial({{color:{arc_color}}});
      const arc=new THREE.Mesh(arcGeo,arcMat);
      arc.rotation.z=Math.PI/2;
      scene.add(arc);
      const pGeo=new THREE.BufferGeometry();
      const pPos=new Float32Array(60*3);
      for(let i=0;i<60;i++){{
        const a=Math.random()*Math.PI*2, r=2.2+Math.random()*.8;
        pPos[i*3]=Math.cos(a)*r; pPos[i*3+1]=Math.sin(a)*r; pPos[i*3+2]=(Math.random()-.5)*1;
      }}
      pGeo.setAttribute('position',new THREE.BufferAttribute(pPos,3));
      const pMat=new THREE.PointsMaterial({{color:{arc_color},size:.06,transparent:true,opacity:.7}});
      scene.add(new THREE.Points(pGeo,pMat));
      let t=0;
      function loop(){{
        requestAnimationFrame(loop); t+=.015;
        arc.rotation.z=Math.PI/2+t*.3;
        renderer.render(scene,camera);
      }}
      loop();
    </script>
    </body></html>
    """, height=170)

    # All-class predictions table
    with st.expander("📊 All class predictions"):
        rows = [
            {
                "Class": format_label(name),
                "Confidence (%)": round(float(raw_preds[i]) * 100, 2),
            }
            for i, name in enumerate(ALL_CLASS_NAMES)
            if not name.startswith("Unknown_class_")
        ]
        df = (
            pd.DataFrame(rows)
            .sort_values("Confidence (%)", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS  (PhytoScan dark sci-fi theme)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Exo+2:wght@300;400;700;900&family=Share+Tech+Mono&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: #020810 !important;
    font-family: 'Exo 2', sans-serif !important;
    overflow-x: hidden;
}

#MainMenu, footer, header { visibility: hidden !important; }
.block-container {
    padding: 0 2rem 4rem 2rem !important;
    max-width: 1400px !important;
    margin: 0 auto;
}

/* Animated mesh background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 60% at 15% 20%, rgba(0,255,120,0.09) 0%, transparent 65%),
        radial-gradient(ellipse 50% 70% at 85% 75%, rgba(0,180,255,0.07) 0%, transparent 65%),
        radial-gradient(ellipse 80% 40% at 50% 50%, rgba(80,0,255,0.04) 0%, transparent 70%),
        radial-gradient(ellipse 30% 30% at 70% 20%, rgba(0,255,200,0.06) 0%, transparent 60%);
    animation: meshPulse 10s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

/* Grid lines */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,255,120,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,120,0.04) 1px, transparent 1px);
    background-size: 70px 70px;
    animation: gridDrift 20s linear infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes meshPulse {
    0%   { opacity:.7; transform:scale(1); }
    50%  { opacity:1;  transform:scale(1.03); }
    100% { opacity:.8; transform:scale(1.01); }
}
@keyframes gridDrift {
    0%   { background-position:0 0; }
    100% { background-position:70px 70px; }
}

/* Upload widget */
[data-testid="stFileUploadDropzone"] {
    background: rgba(0,255,120,0.03) !important;
    border: 2px dashed rgba(0,255,120,0.35) !important;
    border-radius: 20px !important;
    transition: all 0.4s ease !important;
    animation: dropPulse 3s ease-in-out infinite !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(0,255,120,0.07) !important;
    border-color: #00ff80 !important;
    box-shadow: 0 0 50px rgba(0,255,120,0.2), inset 0 0 50px rgba(0,255,120,0.03) !important;
}
@keyframes dropPulse {
    0%,100% { box-shadow:0 0 20px rgba(0,255,120,0.05); }
    50%      { box-shadow:0 0 40px rgba(0,255,120,0.15); }
}
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small {
    color: rgba(0,255,120,0.75) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* Image */
[data-testid="stImage"] img {
    border-radius: 18px !important;
    border: 1px solid rgba(0,255,120,0.25) !important;
    box-shadow: 0 0 50px rgba(0,0,0,0.7), 0 0 25px rgba(0,255,120,0.08) !important;
    transition: all 0.4s ease !important;
}
[data-testid="stImage"] img:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 0 70px rgba(0,0,0,0.8), 0 0 40px rgba(0,255,120,0.2) !important;
}

/* Progress bars */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00ff80, #00cfff, #7c3aed) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 12px rgba(0,255,120,0.6) !important;
}
.stProgress > div > div > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 999px !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #00ff80 !important; }

/* Markdown */
.stMarkdown p { color: rgba(200,220,210,0.8) !important; line-height:1.7; }
.stMarkdown h3 {
    color: #00ff80 !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: .08em;
}

/* Alert overrides */
[data-testid="stAlert"] {
    background: rgba(0,255,120,0.04) !important;
    border: 1px solid rgba(0,255,120,0.25) !important;
    border-radius: 14px !important;
    color: rgba(180,255,200,0.9) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(0,255,120,0.15) !important;
    border-radius: 14px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #020810; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#00ff80, #00cfff);
    border-radius: 3px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(2,8,16,.95) !important;
    border-right: 1px solid rgba(0,255,120,.15) !important;
}

/* Glass card */
.phyto-section {
    position: relative;
    z-index: 2;
    animation: fadeUp .8s cubic-bezier(.22,1,.36,1) both;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(28px); }
    to   { opacity:1; transform:translateY(0); }
}
.phyto-section:nth-child(2) { animation-delay:.15s; }
.phyto-section:nth-child(3) { animation-delay:.3s;  }

.glass {
    background: rgba(255,255,255,0.025);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(0,255,120,0.14);
    border-radius: 24px;
    padding: 2rem 2.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 60px rgba(0,0,0,.5), inset 0 0 60px rgba(0,0,0,.2);
    transition: border-color .4s, box-shadow .4s, transform .4s;
}
.glass:hover {
    border-color: rgba(0,255,120,.3);
    box-shadow: 0 4px 80px rgba(0,0,0,.6), 0 0 40px rgba(0,255,120,.08);
    transform: translateY(-3px);
}
/* scan line */
.glass::after {
    content: '';
    position: absolute;
    top:0; left:-100%;
    width:100%; height:1px;
    background: linear-gradient(90deg,transparent,rgba(0,255,120,.6),transparent);
    animation: scan 4s linear infinite;
}
@keyframes scan { to { left:200%; } }

.neon-divider {
    height: 1px;
    background: linear-gradient(90deg,transparent,rgba(0,255,120,.4),rgba(0,207,255,.4),transparent);
    margin: 1.5rem 0;
    border: none;
}

.big-num {
    font-family: 'Exo 2', sans-serif;
    font-weight: 900;
    font-size: 4rem;
    line-height: 1;
    background: linear-gradient(135deg, #00ff80 0%, #00cfff 60%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 20px rgba(0,255,120,.4));
}

.stat-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: .68rem;
    letter-spacing: .22em;
    color: rgba(150,200,170,.6);
    text-transform: uppercase;
    margin-top: .35rem;
}

.tag {
    display: inline-block;
    padding: .25rem 1rem;
    border-radius: 999px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .72rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    border: 1px solid;
}
.tag-green { color:#00ff80; border-color:rgba(0,255,120,.5); background:rgba(0,255,120,.08); }
.tag-blue  { color:#00cfff; border-color:rgba(0,207,255,.5); background:rgba(0,207,255,.08); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Three.js hero banner
# ─────────────────────────────────────────────────────────────────────────────

components.html("""
<!DOCTYPE html><html><head>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:transparent; overflow:hidden; }
  canvas { display:block; }
  #overlay {
    position:absolute; inset:0;
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    pointer-events:none;
  }
  .title {
    font-family:'Exo 2',sans-serif; font-weight:900;
    font-size:clamp(2.4rem,6vw,5rem); letter-spacing:.06em;
    background:linear-gradient(135deg,#00ff80 0%,#00cfff 45%,#a855f7 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    filter:drop-shadow(0 0 30px rgba(0,255,120,.5));
    animation:titlePulse 3s ease-in-out infinite alternate;
  }
  .sub {
    font-family:'Share Tech Mono',monospace;
    font-size:clamp(.65rem,1.5vw,.9rem); letter-spacing:.35em;
    color:rgba(0,255,120,.55); text-transform:uppercase;
    margin-top:.6rem; animation:subFade 4s ease-in-out infinite alternate;
  }
  .pills { display:flex; gap:.8rem; margin-top:1.2rem; flex-wrap:wrap; justify-content:center; }
  .pill {
    font-family:'Share Tech Mono',monospace; font-size:.62rem; letter-spacing:.15em;
    padding:.3rem .9rem; border-radius:999px; border:1px solid rgba(0,255,120,.3);
    color:rgba(0,255,120,.7); background:rgba(0,255,120,.06); text-transform:uppercase;
    animation:pillFloat 3s ease-in-out infinite alternate;
  }
  .pill:nth-child(2){animation-delay:.4s;border-color:rgba(0,207,255,.3);color:rgba(0,207,255,.7);background:rgba(0,207,255,.06);}
  .pill:nth-child(3){animation-delay:.8s;border-color:rgba(168,85,247,.3);color:rgba(168,85,247,.7);background:rgba(168,85,247,.06);}
  @keyframes titlePulse{0%{filter:drop-shadow(0 0 20px rgba(0,255,120,.4));}100%{filter:drop-shadow(0 0 50px rgba(0,207,255,.7));}}
  @keyframes subFade{0%{opacity:.4;}100%{opacity:.8;}}
  @keyframes pillFloat{0%{transform:translateY(0);}100%{transform:translateY(-5px);}}
  @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@900&family=Share+Tech+Mono&display=swap');
</style></head>
<body>
<canvas id="c"></canvas>
<div id="overlay">
  <div class="title">PHYTO SCAN AI</div>
  <div class="sub">Neural Plant Disease Detection · v3.0</div>
  <div class="pills">
    <span class="pill">🌶 Bell Pepper</span>
    <span class="pill">🥔 Potato</span>
    <span class="pill">🍅 Tomato</span>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const W=window.innerWidth, H=window.innerHeight;
const renderer=new THREE.WebGLRenderer({canvas:document.getElementById('c'),alpha:true,antialias:true});
renderer.setSize(W,H); renderer.setPixelRatio(Math.min(devicePixelRatio,2));
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(60,W/H,0.1,1000);
camera.position.z=30;
const COUNT=600, geo=new THREE.BufferGeometry();
const pos=new Float32Array(COUNT*3), col=new Float32Array(COUNT*3), vel=new Float32Array(COUNT*3);
const palettes=[[0,1,0.47],[0,0.81,1],[0.66,0.33,0.98]];
for(let i=0;i<COUNT;i++){
  pos[i*3]=(Math.random()-.5)*120; pos[i*3+1]=(Math.random()-.5)*80; pos[i*3+2]=(Math.random()-.5)*60;
  vel[i*3]=(Math.random()-.5)*.02; vel[i*3+1]=(Math.random()-.5)*.02; vel[i*3+2]=(Math.random()-.5)*.01;
  const c=palettes[Math.floor(Math.random()*palettes.length)];
  col[i*3]=c[0]; col[i*3+1]=c[1]; col[i*3+2]=c[2];
}
geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
geo.setAttribute('color',new THREE.BufferAttribute(col,3));
const mat=new THREE.PointsMaterial({size:.35,vertexColors:true,transparent:true,opacity:.75});
const pts=new THREE.Points(geo,mat); scene.add(pts);
function makeRing(radius,y,color,opacity){
  const rGeo=new THREE.RingGeometry(radius,radius+.12,80);
  const rMat=new THREE.MeshBasicMaterial({color,transparent:true,opacity,side:THREE.DoubleSide});
  const mesh=new THREE.Mesh(rGeo,rMat); mesh.position.y=y; mesh.rotation.x=Math.PI/2; return mesh;
}
const rings=[], ringColors=[0x00ff80,0x00cfff,0xa855f7,0x00ff80,0x00cfff];
for(let i=0;i<5;i++){const r=makeRing(6+i*2.5,(i-2)*6,ringColors[i],.25);scene.add(r);rings.push(r);}
function makeHex(size,p,color){
  const g=new THREE.CylinderGeometry(size,size,.05,6);
  const m=new THREE.MeshBasicMaterial({color,transparent:true,opacity:.15,wireframe:true});
  const h=new THREE.Mesh(g,m); h.position.set(...p); return h;
}
const hexes=[makeHex(4,[-18,5,-10],0x00ff80),makeHex(3,[20,-6,-8],0x00cfff),makeHex(5,[8,10,-15],0xa855f7)];
hexes.forEach(h=>scene.add(h));
let t=0;
function animate(){
  requestAnimationFrame(animate); t+=.008;
  const p=geo.attributes.position.array;
  for(let i=0;i<COUNT;i++){
    p[i*3]+=vel[i*3]; p[i*3+1]+=vel[i*3+1]+Math.sin(t+i)*.003; p[i*3+2]+=vel[i*3+2];
    if(Math.abs(p[i*3])>60)vel[i*3]*=-1;
    if(Math.abs(p[i*3+1])>40)vel[i*3+1]*=-1;
    if(Math.abs(p[i*3+2])>30)vel[i*3+2]*=-1;
  }
  geo.attributes.position.needsUpdate=true;
  pts.rotation.y=t*.05;
  rings.forEach((r,i)=>{r.rotation.z=t*(.3+i*.05);r.position.y=(i-2)*6+Math.sin(t+i)*1.5;r.material.opacity=.15+Math.sin(t*2+i)*.1;});
  hexes.forEach((h,i)=>{h.rotation.y=t*(.2+i*.1);h.rotation.x=t*.1;h.position.y+=Math.sin(t+i*2)*.02;});
  camera.position.x=Math.sin(t*.2)*3; camera.position.y=Math.cos(t*.15)*2; camera.lookAt(0,0,0);
  renderer.render(scene,camera);
}
animate();
</script></body></html>
""", height=340)

# ─────────────────────────────────────────────────────────────────────────────
# Stats row
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="phyto-section" style="margin-top:-12px;">
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1.2rem;">
  <div class="glass" style="text-align:center;padding:1.4rem;">
    <div class="big-num" style="font-size:2.8rem;">11</div>
    <div class="stat-label">Disease Classes</div>
  </div>
  <div class="glass" style="text-align:center;padding:1.4rem;">
    <div class="big-num" style="font-size:2.8rem;">224</div>
    <div class="stat-label">Input Resolution</div>
  </div>
  <div class="glass" style="text-align:center;padding:1.4rem;">
    <div class="big-num" style="font-size:2.8rem;">6</div>
    <div class="stat-label">Validation Signals</div>
  </div>
  <div class="glass" style="text-align:center;padding:1.4rem;">
    <div class="big-num" style="font-size:2.8rem;">75%</div>
    <div class="stat-label">Confidence Gate</div>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar  (debug panel — from original app.py)
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.2rem;
                color:#00cfff;letter-spacing:.1em;margin-bottom:1rem;">
      ⚙️ PHYTO SCAN AI
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
**Pipeline:**

1. 🔍 **Leaf Validation** (6 signals)  
   Skin · Sky · Saturation · Leaf hue · Edges · Texture

2. 🤖 **Disease Model**  
   MobileNetV2 — 11 disease classes

3. 📊 **Confidence Gate**  
   Result shown only if confidence ≥ **75%**

---
**Supported classes:**
- 🌶 Pepper: Bacterial Spot, Healthy  
- 🥔 Potato: Early Blight, Late Blight, Healthy  
- 🍅 Tomato: 7 diseases + Healthy
""")
    show_debug = st.checkbox("🔬 Show validation debug info", value=False)

# ─────────────────────────────────────────────────────────────────────────────
# Upload section
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="phyto-section glass" style="margin-bottom:0;">
  <div style="font-family:'Rajdhani',sans-serif;font-size:1.5rem;font-weight:700;
              color:#00ff80;letter-spacing:.08em;margin-bottom:.3rem;">
    📡 SCAN LEAF SPECIMEN
  </div>
  <p style="font-family:'Share Tech Mono',monospace;font-size:.75rem;
            color:rgba(0,255,120,.5);letter-spacing:.15em;margin-bottom:1.5rem;">
    UPLOAD IMAGE → LEAF VALIDATION → DISEASE ANALYSIS → DIAGNOSIS
  </p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
    help="Tomato, Potato, or Bell Pepper leaf only",
)

# ─────────────────────────────────────────────────────────────────────────────
# Main two-column layout
# ─────────────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    if uploaded:
        pil_image = Image.open(uploaded)
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        st.image(pil_image, use_container_width=True)
        w, h = pil_image.size
        st.markdown(f"""
        <div style="display:flex;gap:.8rem;margin-top:.8rem;flex-wrap:wrap;">
          <span class="tag tag-blue">📐 {w}×{h}px</span>
          <span class="tag tag-green">🖼 {pil_image.mode}</span>
          <span class="tag tag-blue">{uploaded.name}</span>
        </div>
        """, unsafe_allow_html=True)

        if show_debug:
            with st.expander("🔬 Validation signal values", expanded=True):
                st.json(get_validation_debug_info(pil_image))

with col_right:
    if uploaded is None:
        # Awaiting specimen animation
        components.html("""
        <!DOCTYPE html><html><head>
        <style>
          *{margin:0;padding:0;box-sizing:border-box;}
          body{background:transparent;overflow:hidden;display:flex;align-items:center;justify-content:center;height:100%;}
          canvas{position:absolute;inset:0;}
          .msg{position:relative;z-index:2;text-align:center;font-family:'Share Tech Mono',monospace;}
          .icon{font-size:4rem;animation:bounce 2s ease-in-out infinite;}
          .line1{color:rgba(0,255,120,.7);font-size:.85rem;letter-spacing:.2em;margin-top:1rem;}
          .line2{color:rgba(0,255,120,.35);font-size:.65rem;letter-spacing:.15em;margin-top:.5rem;}
          @keyframes bounce{0%,100%{transform:translateY(0) scale(1);}50%{transform:translateY(-12px) scale(1.05);}}
          @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
        </style></head>
        <body>
        <canvas id="c"></canvas>
        <div class="msg">
          <div class="icon">🌿</div>
          <div class="line1">AWAITING SPECIMEN</div>
          <div class="line2">Upload a leaf image to begin neural scan</div>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
          const c=document.getElementById('c');
          const W=window.innerWidth, H=window.innerHeight;
          const renderer=new THREE.WebGLRenderer({canvas:c,alpha:true,antialias:true});
          renderer.setSize(W,H);
          const scene=new THREE.Scene();
          const camera=new THREE.PerspectiveCamera(60,W/H,.1,200);
          camera.position.z=18;
          const geo=new THREE.TorusGeometry(6,.08,8,80);
          const mat=new THREE.MeshBasicMaterial({color:0x00ff80,transparent:true,opacity:.5});
          const torus=new THREE.Mesh(geo,mat); scene.add(torus);
          const geo2=new THREE.TorusGeometry(8,.05,8,80);
          const mat2=new THREE.MeshBasicMaterial({color:0x00cfff,transparent:true,opacity:.3});
          const torus2=new THREE.Mesh(geo2,mat2); scene.add(torus2);
          const dGeo=new THREE.BufferGeometry();
          const dPos=new Float32Array(200*3);
          for(let i=0;i<200;i++){
            const angle=Math.random()*Math.PI*2, r=5+Math.random()*6;
            dPos[i*3]=Math.cos(angle)*r; dPos[i*3+1]=(Math.random()-.5)*12; dPos[i*3+2]=Math.sin(angle)*r;
          }
          dGeo.setAttribute('position',new THREE.BufferAttribute(dPos,3));
          const dMat=new THREE.PointsMaterial({color:0x00ff80,size:.15,transparent:true,opacity:.6});
          scene.add(new THREE.Points(dGeo,dMat));
          let t=0;
          function loop(){
            requestAnimationFrame(loop); t+=.012;
            torus.rotation.x=t*.4; torus.rotation.y=t*.3;
            torus2.rotation.x=-t*.25; torus2.rotation.z=t*.35;
            camera.position.x=Math.sin(t*.3)*3; camera.position.y=Math.cos(t*.2)*2;
            camera.lookAt(0,0,0); renderer.render(scene,camera);
          }
          loop();
        </script></body></html>
        """, height=440)

    else:
        pil_image = Image.open(uploaded)
        model     = load_model()

        if model is None:
            st.markdown("""
            <div class="glass" style="text-align:center;padding:2rem;">
              <div style="font-size:2rem;margin-bottom:1rem;">⚠️</div>
              <div style="font-family:'Rajdhani',sans-serif;color:#ff4d6d;font-size:1.2rem;letter-spacing:.08em;">
                MODEL NOT FOUND
              </div>
              <p style="color:rgba(255,100,100,.6);font-family:'Share Tech Mono',monospace;font-size:.75rem;margin-top:.8rem;">
                Place plant_disease_model.keras in the app directory
              </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # ── Stage 1: Leaf validation (original app.py pipeline) ───────────
            bar = st.progress(0, text="Stage 1 / 3 — Leaf validation…")
            time.sleep(0.2)

            is_leaf, reason = validate_leaf_image(pil_image)
            bar.progress(33, text="Stage 1 / 3 — Leaf validation complete")
            time.sleep(0.15)

            if not is_leaf:
                bar.empty()
                render_result({
                    "status":  "invalid_leaf",
                    "message": "❌ Invalid image.",
                    "detail":  reason,
                })
                st.stop()

            # ── Stage 2: Model inference ──────────────────────────────────────
            bar.progress(45, text="Stage 2 / 3 — Running disease model…")
            raw_preds  = model.predict(preprocess(pil_image), verbose=0)[0]
            top_idx    = int(np.argmax(raw_preds))
            confidence = float(raw_preds[top_idx])
            disease    = ALL_CLASS_NAMES[top_idx]

            bar.progress(75, text="Stage 3 / 3 — Confidence gate…")
            time.sleep(0.15)

            # ── Stage 3: Unknown class check ──────────────────────────────────
            if disease.startswith("Unknown_class_"):
                bar.empty()
                render_result({
                    "status":  "unknown_class",
                    "message": "⚠️ Unable to identify disease.",
                    "detail":  (
                        f"Model output index {top_idx} has no known label. "
                        f"Confidence was {confidence*100:.1f}%. "
                        "Try a clearer image of a Tomato, Potato, or Bell Pepper leaf."
                    ),
                })
                st.stop()

            # ── Stage 4: Confidence gate ──────────────────────────────────────
            if confidence < CONFIDENCE_THRESHOLD:
                bar.empty()
                render_result({
                    "status":     "low_confidence",
                    "message":    "⚠️ Low confidence. Please upload a clearer leaf image.",
                    "detail":     (
                        f"Best match: '{format_label(disease)}' "
                        f"at {confidence*100:.1f}% — below the {CONFIDENCE_THRESHOLD*100:.0f}% threshold."
                    ),
                    "confidence": confidence,
                })
                st.stop()

            bar.progress(100, text="Done ✓")
            time.sleep(0.2)
            bar.empty()

            # ── Show result ───────────────────────────────────────────────────
            info = DISEASE_INFO.get(disease, {
                "crop":        disease.split(" ")[0].title(),
                "common_name": format_label(disease),
                "severity":    "Unknown",
                "treatment":   "Consult a local agricultural extension service.",
            })

            render_result({
                "status":     "valid",
                "disease":    disease,
                "confidence": confidence,
                "info":       info,
                "raw_preds":  raw_preds,
            })

# ─────────────────────────────────────────────────────────────────────────────
# Disease catalogue grid (bottom)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="phyto-section glass">
  <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.2rem;
              color:#00cfff;letter-spacing:.1em;margin-bottom:1.2rem;">
    🗂 DETECTABLE DISEASE CATALOGUE
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:.7rem;">
    <div style="padding:.6rem 1rem;border:1px solid rgba(255,77,109,.2);border-radius:10px;
                background:rgba(255,77,109,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(255,150,150,.7);letter-spacing:.06em;">
      🌶 Pepper — Bacterial Spot
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(0,255,120,.2);border-radius:10px;
                background:rgba(0,255,120,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(150,255,150,.7);letter-spacing:.06em;">
      🌶 Pepper — Healthy
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(255,170,0,.2);border-radius:10px;
                background:rgba(255,170,0,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(255,200,100,.7);letter-spacing:.06em;">
      🥔 Potato — Early Blight
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(255,77,109,.2);border-radius:10px;
                background:rgba(255,77,109,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(255,150,150,.7);letter-spacing:.06em;">
      🥔 Potato — Late Blight
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(0,255,120,.2);border-radius:10px;
                background:rgba(0,255,120,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(150,255,150,.7);letter-spacing:.06em;">
      🥔 Potato — Healthy
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(255,77,109,.2);border-radius:10px;
                background:rgba(255,77,109,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(255,150,150,.7);letter-spacing:.06em;">
      🍅 Tomato — Bacterial Spot
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(255,170,0,.2);border-radius:10px;
                background:rgba(255,170,0,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(255,200,100,.7);letter-spacing:.06em;">
      🍅 Tomato — Early Blight
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(255,77,109,.2);border-radius:10px;
                background:rgba(255,77,109,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(255,150,150,.7);letter-spacing:.06em;">
      🍅 Tomato — Late Blight
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(168,85,247,.2);border-radius:10px;
                background:rgba(168,85,247,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(200,150,255,.7);letter-spacing:.06em;">
      🍅 Tomato — Leaf Mold
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(255,170,0,.2);border-radius:10px;
                background:rgba(255,170,0,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(255,200,100,.7);letter-spacing:.06em;">
      🍅 Tomato — Septoria Leaf Spot
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(255,77,109,.2);border-radius:10px;
                background:rgba(255,77,109,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(255,150,150,.7);letter-spacing:.06em;">
      🍅 Tomato — Target Spot
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(168,85,247,.2);border-radius:10px;
                background:rgba(168,85,247,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(200,150,255,.7);letter-spacing:.06em;">
      🍅 Tomato — Mosaic Virus
    </div>
    <div style="padding:.6rem 1rem;border:1px solid rgba(0,255,120,.2);border-radius:10px;
                background:rgba(0,255,120,.04);font-family:'Share Tech Mono',monospace;
                font-size:.7rem;color:rgba(150,255,150,.7);letter-spacing:.06em;">
      🍅 Tomato — Healthy
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;margin-top:2rem;
     font-family:'Share Tech Mono',monospace;font-size:.62rem;
     letter-spacing:.2em;color:rgba(0,255,120,.25);">
  PHYTO SCAN AI · 6-SIGNAL LEAF VALIDATOR · NEURAL PATHOGEN DETECTION · POWERED BY KERAS + TENSORFLOW
</div>
""", unsafe_allow_html=True)
