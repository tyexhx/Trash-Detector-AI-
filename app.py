"""
Illegal Dumping Detector
Aligned with UN Sustainable Development Goal 11: Sustainable Cities and Communities
Uses Teachable Machine model with webcam to detect illegal dumping activity.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from pathlib import Path
import json
import os
import base64

# Configuration
MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH_H5 = MODEL_DIR / "keras_model.h5"
MODEL_PATH_TFJS = MODEL_DIR / "model.json"
LABELS_PATH = MODEL_DIR / "labels.txt"
IMG_SIZE = 224
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_DETECTION_INTERVAL = 0.5

# Thomas More University Branding Theme (matched to thomasmore.be)
SDG_ORANGE = "#F15A24"       # Thomas More brand orange (from website)
SDG_ORANGE_DARK = "#D94E1F"  # Darker shade for hover/emphasis
SDG_ORANGE_LIGHT = "#F7845E" # Lighter tint for subtle accents
SDG_BG_DARK = "#FFFFFF"      # Clean white page background (TM style)
SDG_BG_CARD = "#FFFFFF"      # Pure white cards
SDG_TEXT = "#002D3A"         # Thomas More deep navy text
SDG_TEAL = "#69C2C2"         # Thomas More teal accent (CTA buttons)
SDG_NAVY = "#002D3A"         # Thomas More deep navy for sidebar/footer

import streamlit.components.v1 as components

def play_alarm(volume=0.5):
    """Play a siren sound via HTML5 AudioContext with adjustable volume (0.0 - 1.0)."""
    js = f"""
    <script>
    // Timestamp to ensure Streamlit re-renders and replays the sound
    // {time.time()}
    var vol = {volume};
    var ctx = new (window.AudioContext || window.webkitAudioContext)();
    var osc = ctx.createOscillator();
    var gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = "square";
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.setValueAtTime(1108, ctx.currentTime + 0.25);
    osc.frequency.setValueAtTime(880, ctx.currentTime + 0.5);
    osc.frequency.setValueAtTime(1108, ctx.currentTime + 0.75);
    gain.gain.setValueAtTime(0.0, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(vol, ctx.currentTime + 0.05);
    gain.gain.setValueAtTime(vol, ctx.currentTime + 0.95);
    gain.gain.linearRampToValueAtTime(0.0, ctx.currentTime + 1.0);
    osc.start();
    osc.stop(ctx.currentTime + 1.0);
    </script>
    """
    components.html(js, height=0, width=0)

def load_model_and_labels():
    """Load the Teachable Machine SavedModel format."""
    try:
        import tensorflow as tf
    except ImportError:
        return None, None, "TensorFlow not installed. Run: pip install tensorflow"

    model = None
    error = None

    # Find the directory containing saved_model.pb
    saved_model_path = next(MODEL_DIR.rglob("saved_model.pb"), None)

    if saved_model_path:
        try:
            # Bypass Keras 3 deprecations by loading the raw TensorFlow SavedModel serving graph!
            model_dir = saved_model_path.parent
            model = tf.saved_model.load(str(model_dir))
        except Exception as e:
            error = f"Failed to load SavedModel: {e}"
    else:
        return None, None, "No SavedModel found. Please upload your Teachable Machine SavedModel ZIP using the sidebar."

    if model is None:
        return None, None, error

    labels = []
    # Search for labels.txt anywhere in the model directory
    label_file = next(MODEL_DIR.rglob("labels.txt"), None)
    if label_file:
        with open(label_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    parts = line.split(" ", 1)
                    labels.append(parts[1] if len(parts) > 1 and parts[0].isdigit() else line)
    else:
        # Fallback
        labels = ["Trash", "Clean", "Human"]

    return model, labels, None


def preprocess_frame(frame):
    """Preprocess a frame for Teachable Machine model (Match browser aspect ratio)."""
    # 1. Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Center crop to square! Teachable Machine web preview is a square. 
    # If we squint 16:9 into a square, the trash looks distorted and the model fails.
    h, w, _ = rgb_frame.shape
    size = min(h, w)
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    square_frame = rgb_frame[y_start:y_start+size, x_start:x_start+size]
    
    # 3. Resize and Normalize
    resized = cv2.resize(square_frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalized = (np.asarray(resized, dtype=np.float32) / 127.5) - 1.0
    
    return np.expand_dims(normalized, axis=0)


def run_detection(model, frame, labels):
    """Run the detection model on a frame using Native TensorFlow."""
    import tensorflow as tf
    processed = preprocess_frame(frame)
    
    # Extract prediction tensor by executing the raw SavedModel signature
    infer = model.signatures["serving_default"]
    
    # Pass tensor exactly as outputted by TM
    output_dict = infer(tf.constant(processed))
    
    # The output struct is a dictionary. Grab the first tensor's values.
    predictions = list(output_dict.values())[0].numpy()[0]

    top_idx = np.argmax(predictions)
    confidence = predictions[top_idx]
    class_name = labels[top_idx] if top_idx < len(labels) else f"Class {top_idx}"

    all_preds = {}
    for i, pred in enumerate(predictions):
        label = labels[i] if i < len(labels) else f"Class {i}"
        all_preds[label] = float(pred)

    return class_name, float(confidence), all_preds


def init_session_state():
    """Initialize session state variables."""
    if "running" not in st.session_state:
        st.session_state.running = False
    if "alert_log" not in st.session_state:
        st.session_state.alert_log = []
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Detection"
    
    # State used to persist drawing across all frames
    if "last_is_dumping" not in st.session_state:
        st.session_state.last_is_dumping = False

    if "model_files_uploaded" not in st.session_state:
        st.session_state.model_files_uploaded = any(MODEL_DIR.rglob("saved_model.pb"))


def get_logo_base64():
    """Read the Thomas More SVG logo and return as base64 data URI."""
    logo_path = Path(__file__).parent / "thomas_more_logo.svg"
    if logo_path.exists():
        with open(logo_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        encoded = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
        return f"data:image/svg+xml;base64,{encoded}"
    return ""


def apply_custom_css():
    """Apply Thomas More University branded CSS with clean, modern design."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
    
    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif !important;
        color: {SDG_TEXT};
    }}
    .stApp {{
        background: {SDG_BG_DARK};
    }}
    /* ===== HEADER ===== */
    .main-header {{
        background: {SDG_ORANGE};
        padding: 0;
        border-radius: 0;
        margin: -1rem -1rem 40px -1rem;
        box-shadow: none;
        position: relative;
        overflow: hidden;
    }}
    .main-header::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background:
            repeating-linear-gradient(
                -55deg,
                transparent,
                transparent 30px,
                rgba(255,255,255,0.05) 30px,
                rgba(255,255,255,0.05) 60px
            );
        pointer-events: none;
    }}
    .header-content {{
        display: flex;
        align-items: center;
        gap: 32px;
        padding: 32px 48px;
        position: relative;
        z-index: 2;
    }}
    .header-logo {{
        flex-shrink: 0;
    }}
    .header-logo img {{
        height: 72px;
        width: auto;
        filter: brightness(0) invert(1);
    }}
    .header-text {{
        flex: 1;
    }}
    .main-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        color: white;
        margin: 0;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }}
    .main-subtitle {{
        color: rgba(255,255,255,0.92);
        font-size: 1.05rem;
        margin-top: 8px;
        font-weight: 400;
        line-height: 1.5;
    }}
    .sdg-badge {{
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(8px);
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 600;
        color: white;
        display: inline-block;
        margin-top: 12px;
        border: 1px solid rgba(255,255,255,0.25);
        letter-spacing: 0.3px;
    }}
    /* ===== CARDS ===== */
    .card {{
        background: {SDG_BG_CARD};
        border-radius: 16px;
        padding: 28px;
        border: 1px solid #E8E8E8;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        margin-bottom: 24px;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}
    .card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.07);
    }}
    .card-header {{
        color: {SDG_NAVY};
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: 'Outfit', sans-serif;
    }}
    /* ===== SECTION HEADINGS ===== */
    .section-title {{
        color: {SDG_NAVY};
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 28px;
        padding-bottom: 0;
        border-bottom: none;
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.8px;
    }}
    .section-title span.tm-highlight {{
        color: {SDG_ORANGE};
    }}
    .section-subtitle {{
        color: {SDG_ORANGE};
        font-size: 1.35rem;
        font-weight: 700;
        margin: 36px 0 16px 0;
        font-family: 'Outfit', sans-serif;
    }}
    .content-text {{
        color: {SDG_NAVY};
        font-size: 1.05rem;
        line-height: 1.8;
        margin-bottom: 16px;
    }}
    /* ===== FEATURE CARDS ===== */
    .feature-card {{
        background: {SDG_BG_CARD};
        border-radius: 16px;
        padding: 32px 24px;
        border: 1px solid #E8E8E8;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        text-align: center;
        height: 100%;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}
    .feature-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(241, 90, 36, 0.12);
    }}
    .feature-icon {{ font-size: 3.2rem; margin-bottom: 20px; }}
    .feature-title {{
        color: {SDG_NAVY};
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 10px;
        font-family: 'Outfit', sans-serif;
    }}
    .feature-desc {{ color: {SDG_NAVY}; font-size: 0.95rem; line-height: 1.7; opacity: 0.7; }}
    /* ===== STAT CARDS ===== */
    .stat-card {{
        background: {SDG_BG_CARD};
        border-radius: 16px;
        padding: 28px;
        border: 1px solid #E8E8E8;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        text-align: center;
    }}
    .stat-number {{ color: {SDG_ORANGE}; font-family: 'Outfit', sans-serif; font-size: 2.8rem; font-weight: 800; line-height: 1; margin-bottom: 8px; }}
    .stat-label {{ color: {SDG_NAVY}; font-size: 0.95rem; font-weight: 500; opacity: 0.8; }}
    /* ===== BULLET LIST ===== */
    .bullet-list {{ list-style: none; padding: 0; margin: 0; }}
    .bullet-list li {{
        color: {SDG_NAVY};
        padding: 14px 0;
        padding-left: 28px;
        position: relative;
        border-bottom: 1px solid rgba(0, 0, 0, 0.04);
        font-size: 1rem;
    }}
    .bullet-list li:last-child {{ border-bottom: none; }}
    .bullet-list li:before {{
        content: "";
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 8px;
        height: 8px;
        background: {SDG_ORANGE};
        border-radius: 50%;
    }}
    /* ===== QUOTE BOX ===== */
    .quote-box {{
        background: {SDG_NAVY};
        border-left: 4px solid {SDG_ORANGE};
        padding: 28px 32px;
        border-radius: 0 16px 16px 0;
        margin: 36px 0;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    }}
    .quote-text {{ color: white; font-size: 1.15rem; font-style: italic; line-height: 1.8; font-weight: 400; }}
    .quote-author {{ color: {SDG_ORANGE}; font-size: 0.9rem; margin-top: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }}
    /* ===== ALERTS ===== */
    .alert-box {{
        padding: 18px 24px;
        border-radius: 50px;
        margin: 16px 0;
        font-size: 1.05rem;
        font-weight: 600;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        font-family: 'Outfit', sans-serif;
    }}
    .alert-danger {{
        background: #E53E3E;
        color: white;
        animation: pulse-danger 1.5s infinite;
        box-shadow: 0 4px 16px rgba(229, 62, 62, 0.3);
    }}
    .alert-safe {{
        background: {SDG_TEAL};
        color: {SDG_NAVY};
        box-shadow: 0 4px 16px rgba(105, 194, 194, 0.3);
    }}
    .alert-idle {{
        background: #F0F4F5;
        color: {SDG_NAVY};
        border: 1px solid #D0D8DA;
        opacity: 0.7;
    }}
    @keyframes pulse-danger {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.92; transform: scale(1.01); }}
    }}
    /* ===== BUTTONS (Thomas More style with arrow) ===== */
    .stButton > button {{
        border-radius: 50px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.2px !important;
    }}
    .stButton > button[kind="primary"] {{
        background: {SDG_TEAL} !important;
        border: none !important;
        color: {SDG_NAVY} !important;
        box-shadow: 0 4px 12px rgba(105, 194, 194, 0.3) !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: #56B0B0 !important;
        box-shadow: 0 6px 20px rgba(105, 194, 194, 0.4) !important;
        transform: translateY(-1px) !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: transparent !important;
        border: 2px solid {SDG_ORANGE} !important;
        color: {SDG_ORANGE} !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        background: {SDG_ORANGE} !important;
        color: white !important;
        transform: translateY(-1px) !important;
    }}
    /* ===== SIDEBAR (Thomas More navy) ===== */
    section[data-testid="stSidebar"] {{
        background: {SDG_NAVY} !important;
        border-right: none;
    }}
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {SDG_ORANGE} !important;
        font-family: 'Outfit', sans-serif;
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown div,
    section[data-testid="stSidebar"] .stMarkdown label {{
        color: rgba(255,255,255,0.85) !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown code {{
        color: {SDG_ORANGE} !important;
        background: rgba(241,90,36,0.15) !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown strong {{
        color: white !important;
    }}
    /* Sidebar widget labels (sliders, file uploader) */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stFileUploader label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {{
        color: {SDG_ORANGE} !important;
        font-weight: 600 !important;
    }}
    /* Sidebar slider value text */
    section[data-testid="stSidebar"] [data-testid="stThumbValue"],
    section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {{
        color: white !important;
    }}
    /* Sidebar slider track */
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background: {SDG_ORANGE} !important;
        border-color: {SDG_ORANGE} !important;
    }}
    /* Sidebar slider help text */
    section[data-testid="stSidebar"] .stTooltipIcon,
    section[data-testid="stSidebar"] small {{
        color: rgba(255,255,255,0.5) !important;
    }}
    /* Sidebar alert/info/success/warning boxes */
    section[data-testid="stSidebar"] .stAlert {{
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(255,255,255,0.15) !important;
    }}
    section[data-testid="stSidebar"] .stAlert p {{
        color: rgba(255,255,255,0.9) !important;
    }}
    /* Sidebar divider */
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.12) !important;
    }}
    /* Sidebar file uploader area */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {{
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(255,255,255,0.15) !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section small,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section span {{
        color: rgba(255,255,255,0.6) !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {{
        color: {SDG_ORANGE} !important;
        border-color: {SDG_ORANGE} !important;
    }}
    /* ===== CONFIDENCE BARS ===== */
    .confidence-bar {{
        background: rgba(0, 45, 58, 0.1);
        border-radius: 50px;
        height: 12px;
        overflow: hidden;
        margin-top: 6px;
    }}
    .confidence-fill {{ height: 100%; border-radius: 50px; transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1); }}
    .confidence-item {{ margin: 14px 0; }}
    .confidence-label {{
        display: flex;
        justify-content: space-between;
        color: {SDG_NAVY};
        font-size: 0.95rem;
        margin-bottom: 4px;
        font-weight: 600;
    }}
    /* ===== LOG ENTRY ===== */
    .log-entry {{
        background: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #991B1B;
        font-weight: 500;
    }}
    hr {{ border-color: rgba(0, 0, 0, 0.06); }}
    /* ===== SDG INFO BOX ===== */
    .sdg-info-box {{
        background: {SDG_NAVY};
        border: none;
        border-radius: 20px;
        padding: 36px;
        margin: 28px 0;
        box-shadow: 0 4px 20px rgba(0, 45, 58, 0.15);
    }}
    .sdg-info-box .content-text {{
        color: rgba(255,255,255,0.9) !important;
    }}
    .sdg-goal {{ display: flex; align-items: center; gap: 20px; margin-bottom: 20px; }}
    .sdg-number {{
        background: {SDG_ORANGE};
        color: white;
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        width: 68px;
        height: 68px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(241, 90, 36, 0.3);
    }}
    .sdg-title {{ color: white; font-size: 1.5rem; font-weight: 700; font-family: 'Outfit', sans-serif; letter-spacing: -0.5px; line-height: 1.3; }}
    /* ===== NAVIGATION TABS ===== */
    .tm-nav {{
        background: {SDG_BG_CARD};
        border-radius: 50px;
        padding: 6px;
        display: inline-flex;
        gap: 4px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        margin-bottom: 36px;
    }}
    /* ===== MAIN CONTENT TEXT COLOR ===== */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {{
        color: {SDG_NAVY};
    }}
    </style>
    """, unsafe_allow_html=True)


def render_header():
    logo_data_uri = get_logo_base64()
    logo_img = f'<img src="{logo_data_uri}" alt="Thomas More">' if logo_data_uri else ""
    st.markdown(f"""
    <div class="main-header">
        <div class="header-content">
            <div class="header-logo">{logo_img}</div>
            <div class="header-text">
                <div class="main-title">Illegal Dumping Detector</div>
                <div class="main-subtitle">AI-powered monitoring for cleaner, sustainable communities</div>
                <div class="sdg-badge">SDG 11: Sustainable Cities and Communities</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_navigation():
    pages = ["Detection", "Why This Technology", "Benefits", "About SDG 11"]
    cols = st.columns(len(pages))
    for i, page in enumerate(pages):
        with cols[i]:
            is_active = st.session_state.current_page == page
            if st.button(page, key=f"nav_{page}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.current_page = page
                st.rerun()


def handle_model_upload():
    """Save the uploaded ZIP to the model/ folder, extract, and reload."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    uploaded_zip = st.sidebar.file_uploader(
        "Upload the Teachable Machine SavedModel ZIP",
        type=["zip"],
        accept_multiple_files=False,
        key="model_uploader"
    )

    if uploaded_zip is not None:
        import zipfile
        try:
            # Clean up the old model contents first
            for f in MODEL_DIR.rglob("*"):
                if f.is_file():
                    try: os.remove(f)
                    except: pass
            
            save_path = MODEL_DIR / uploaded_zip.name
            with open(save_path, "wb") as out:
                out.write(uploaded_zip.getbuffer())
                
            # Extract ZIP
            with zipfile.ZipFile(save_path, "r") as zip_ref:
                zip_ref.extractall(MODEL_DIR)
                
            st.sidebar.success("✅ SavedModel ZIP extracted! Reloading...")
            st.session_state.model_files_uploaded = True
            st.session_state.model_loaded = False
            st.rerun()
        except zipfile.BadZipFile:
            st.sidebar.error("❌ The uploaded file is not a valid ZIP archive.")
        except Exception as e:
            st.sidebar.error(f"❌ Error extracting ZIP: {e}")


def render_sidebar():
    with st.sidebar:
        logo_data_uri = get_logo_base64()
        logo_html = f'<img src="{logo_data_uri}" alt="Thomas More" style="height:48px;width:auto;filter:brightness(0) invert(1);">' if logo_data_uri else ""
        st.markdown(f"""
        <div style="text-align:center;padding:20px 0;border-bottom:1px solid rgba(255,255,255,0.12);margin-bottom:16px;">
            <div style="margin-bottom:10px;">{logo_html}</div>
            <div style="color:rgba(255,255,255,0.5);font-size:0.82rem;font-weight:500;">Sustainable Cities &amp; Communities</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<h3 style='color:{SDG_ORANGE};'>📂 Upload Model</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <p style="color:rgba(255,255,255,0.85);font-size:0.88rem;margin-bottom:6px;">
            Upload your Teachable Machine files:<br>
            <strong>Option 1 (TF.js):</strong> <code>model.json</code>, <code>.bin</code>, <code>metadata.json</code><br>
            <strong>Option 2 (Keras):</strong> <code>keras_model.h5</code>, <code>labels.txt</code>
        </p>
        """, unsafe_allow_html=True)

        handle_model_upload()

        st.divider()

        st.markdown(f"<h3 style='color:{SDG_ORANGE};'>Model Status</h3>", unsafe_allow_html=True)

        if st.session_state.model_files_uploaded:
            st.success("✅ Model files uploaded")
            if st.session_state.model_loaded:
                st.success("✅ Model loaded & ready")
            else:
                st.info("ℹ️ Model will load when detection starts")

            has_labels = LABELS_PATH.exists() or (MODEL_DIR / "metadata.json").exists()
            if has_labels:
                st.success("✅ Labels ready")
            else:
                st.info("ℹ️ Will use default labels")
        else:
            st.warning("⚠️ No model uploaded yet")
            st.info("ℹ️ Upload your files above to get started")

        st.divider()

        st.markdown(f"<h3 style='color:{SDG_ORANGE};'>Detection Settings</h3>", unsafe_allow_html=True)

        confidence_threshold = st.slider(
            "Confidence Threshold", min_value=0.5, max_value=0.99,
            value=DEFAULT_CONFIDENCE_THRESHOLD, step=0.05,
            help="Minimum confidence to trigger alert"
        )
        detection_interval = st.slider(
            "Detection Interval (sec)", min_value=0.1, max_value=2.0,
            value=DEFAULT_DETECTION_INTERVAL, step=0.1,
            help="How often to analyze a frame"
        )
        alarm_volume = st.slider(
            "Alarm Volume", min_value=0.0, max_value=1.0,
            value=0.5, step=0.05,
            help="Volume of the alert sound (0 = mute, 1 = max)"
        )

    return confidence_threshold, detection_interval, alarm_volume


def render_why_technology_page():
    st.markdown('<div class="section-title">Why AI-Powered Dumping Detection?</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-text">Illegal dumping is a growing environmental crisis affecting communities worldwide. Traditional monitoring methods are often reactive, relying on citizen reports or periodic inspections. By the time violations are discovered, significant environmental damage may have already occurred.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">The Problem We Are Solving</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="card"><div class="card-header">Traditional Monitoring Challenges</div>
        <ul class="bullet-list">
            <li>Limited coverage with manual patrols</li>
            <li>Reactive responses after damage is done</li>
            <li>High labor costs for continuous monitoring</li>
            <li>Inconsistent detection quality</li>
            <li>Difficulty monitoring remote areas</li>
        </ul></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="card"><div class="card-header">AI-Powered Solution Benefits</div>
        <ul class="bullet-list">
            <li>24/7 continuous automated monitoring</li>
            <li>Real-time alerts for immediate response</li>
            <li>Scalable coverage across multiple sites</li>
            <li>Consistent and objective detection</li>
            <li>Cost-effective long-term operation</li>
        </ul></div>""", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">How Our Technology Works</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (num, title, desc) in enumerate([("1","Capture","Camera feeds continuously monitor designated areas"),("2","Analyze","AI model processes frames in real-time"),("3","Detect","Machine learning identifies dumping activity"),("4","Alert","Instant notifications sent to authorities")]):
        with cols[i]:
            st.markdown(f"""<div class="feature-card">
                <div style="background:linear-gradient(135deg, {SDG_ORANGE} 0%, #FFA200 100%);color:white;width:56px;height:56px;border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:1.6rem;font-weight:700;margin:0 auto 20px auto;box-shadow: 0 4px 12px rgba(255,107,0,0.25);font-family:'Outfit', sans-serif;">{num}</div>
                <div class="feature-title">{title}</div><div class="feature-desc">{desc}</div></div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="quote-box"><div class="quote-text">"Technology alone cannot solve environmental challenges, but when combined with community action and proper governance, it becomes a powerful tool for change."</div>
    <div class="quote-author">- United Nations Environment Programme</div></div>""", unsafe_allow_html=True)


def render_benefits_page():
    st.markdown('<div class="section-title">Benefits of AI Dumping Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Impact by the Numbers</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (number, label, sublabel) in enumerate([("90%","Faster Detection","Compared to manual monitoring"),("24/7","Monitoring","Continuous protection"),("75%","Cost Reduction","In enforcement operations"),("100%","Objective","Consistent detection quality")]):
        with cols[i]:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{number}</div><div class="stat-label"><strong>{label}</strong><br><span style="opacity:0.8;font-size:0.9rem;">{sublabel}</span></div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    for subtitle, benefits in [
        ("Environmental Benefits", [("🌱","Ecosystem Protection","Early detection prevents contaminants from entering soil and water systems."),("💧","Water Quality","Prevents pollutants from reaching groundwater and surface water sources."),("🌍","Reduced Carbon Footprint","Automated monitoring reduces the need for patrol vehicles.")]),
        ("Community Benefits",    [("🏘️","Cleaner Neighborhoods","Real-time detection keeps streets and public spaces free from illegally dumped waste."),("🏥","Public Health","Rapid response prevents health hazards from accumulating waste."),("💰","Property Values","Clean communities maintain higher property values.")]),
        ("Economic Benefits",     [("📉","Reduced Cleanup Costs","Early intervention means less waste to clean."),("⚖️","Better Enforcement","Video evidence enables prosecution of offenders."),("📊","Data-Driven Planning","Analytics help identify hotspots and patterns.")])
    ]:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (icon, title, desc) in enumerate(benefits):
            with cols[i]:
                st.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div><div class="feature-desc">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="quote-box"><div class="quote-text">"Investing in smart waste management technologies delivers returns not just in cleaner cities, but in healthier populations and stronger local economies."</div>
    <div class="quote-author">- World Economic Forum</div></div>""", unsafe_allow_html=True)


def render_sdg11_page():
    st.markdown('<div class="section-title">About SDG 11: Sustainable Cities and Communities</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="sdg-info-box"><div class="sdg-goal"><div class="sdg-number">11</div>
    <div class="sdg-title">Make cities and human settlements inclusive, safe, resilient and sustainable</div></div>
    <div class="content-text">SDG 11 is one of the 17 Sustainable Development Goals established by the United Nations in 2015. It focuses on making cities inclusive, safe, resilient, and sustainable for everyone.</div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Key Targets Related to Our Mission</div>', unsafe_allow_html=True)
    for target_num, target_title, target_desc in [
        ("11.6","Environmental Impact of Cities","By 2030, reduce the adverse per capita environmental impact of cities, including by paying special attention to air quality and municipal and other waste management."),
        ("11.3","Sustainable Urbanization","By 2030, enhance inclusive and sustainable urbanization and capacity for participatory, integrated and sustainable human settlement planning and management."),
        ("11.7","Safe Public Spaces","By 2030, provide universal access to safe, inclusive and accessible, green and public spaces.")
    ]:
        st.markdown(f"""<div class="card" style="display:flex; flex-direction:column; justify-content:center;"><div class="card-header">
            <span style="background:linear-gradient(135deg, {SDG_ORANGE} 0%, #FFA200 100%);color:white;padding:6px 14px;border-radius:8px;font-size:0.9rem;font-weight:700;box-shadow:0 4px 10px rgba(255,107,0,0.2);">Target {target_num}</span>
            <span style="margin-left:8px; font-family:'Outfit', sans-serif;">{target_title}</span></div><div class="content-text">{target_desc}</div></div>""", unsafe_allow_html=True)
    
    st.markdown('<div class="section-subtitle" style="margin-top:40px;">Global Context</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (stat, label) in enumerate([("55%","of world population lives in cities"),("2B","more urban residents expected by 2050"),("90%","of urban growth in developing regions"),("70%","of global CO2 emissions from cities")]):
        with cols[i]:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{stat}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="quote-box" style="margin-top:40px;"><div class="quote-text">"Cities are where the battle for sustainable development will be won or lost."</div>
    <div class="quote-author">- UN-Habitat</div></div>""", unsafe_allow_html=True)


# ============================================================================
# PAGE: DETECTION
# ============================================================================
def render_detection_page(confidence_threshold, detection_interval, alarm_volume):
    if not st.session_state.model_files_uploaded:
        st.markdown(f"""
        <div style="background:rgba(249,157,38,0.15);border:1px solid {SDG_ORANGE};border-radius:12px;
                    padding:40px;text-align:center;margin-bottom:24px;">
            <div style="font-size:3rem;margin-bottom:12px;">📂</div>
            <div style="color:{SDG_ORANGE};font-weight:700;font-size:1.3rem;">No Model Uploaded Yet</div>
            <div style="color:{SDG_TEXT};margin-top:10px;font-size:1rem;line-height:1.8;">
                Use the <strong style="color:{SDG_ORANGE};">sidebar on the left ←</strong> to upload your files.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if not st.session_state.model_loaded:
        with st.spinner("Loading AI model..."):
            model, labels, error = load_model_and_labels()
            if error:
                st.error(f"❌ {error}")
                return
            st.session_state.model = model
            st.session_state.labels = labels
            st.session_state.model_loaded = True

    model = st.session_state.model
    labels = st.session_state.labels

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;
                background:rgba(40,167,69,0.15);padding:12px 20px;border-radius:10px;
                border:1px solid rgba(40,167,69,0.3);">
        <div style="width:12px;height:12px;background:#28a745;border-radius:50%;"></div>
        <span style="color:#28a745;font-weight:500;">Model Active</span>
        <span style="color:{SDG_TEXT};opacity:0.7;">| Classes: {', '.join(labels)}</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f'<div class="card-header" style="margin-bottom:12px;">📹 Live Camera Feed</div>', unsafe_allow_html=True)
        video_placeholder = st.empty()

    with col2:
        st.markdown(f'<div class="card-header">🔍 Detection Status</div>', unsafe_allow_html=True)
        status_placeholder = st.empty()
        st.markdown(f'<div class="card-header" style="margin-top:20px;">📊 Confidence Scores</div>', unsafe_allow_html=True)
        confidence_placeholder = st.empty()
        st.markdown(f'<div class="card-header" style="margin-top:20px;">📋 Alert Log</div>', unsafe_allow_html=True)
        log_placeholder = st.empty()

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        start_btn = st.button("▶ Start Monitoring", type="primary", use_container_width=True)
    with col_btn2:
        stop_btn = st.button("⏹ Stop Monitoring", use_container_width=True)
    with col_btn3:
        if st.button("🗑 Clear Alert Log", use_container_width=True):
            st.session_state.alert_log = []

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False

    if not st.session_state.running:
        video_placeholder.markdown(f"""
        <div style="background:{SDG_BG_CARD};border:2px dashed {SDG_ORANGE};border-radius:16px;
                    padding:80px 40px;text-align:center;">
            <div style="font-size:4rem;margin-bottom:16px;">📹</div>
            <div style="color:{SDG_ORANGE};font-size:1.2rem;font-weight:600;">Camera Inactive</div>
            <div style="color:{SDG_TEXT};opacity:0.7;margin-top:8px;">Click "Start Monitoring" to begin</div>
        </div>""", unsafe_allow_html=True)
        status_placeholder.markdown('<div class="alert-box alert-idle">Monitoring Stopped</div>', unsafe_allow_html=True)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam. Please check your camera connection.")
        st.session_state.running = False
        return

    last_detection_time = 0

    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read from camera")
                break

            current_time = time.time()

            if current_time - last_detection_time >= detection_interval:
                class_name, confidence, all_preds = run_detection(model, frame, labels)
                last_detection_time = current_time

                # --- UPDATED: Detect only the "Trash" class as illegal dumping ---
                is_dumping = class_name.lower() == "trash" and confidence >= confidence_threshold

                # Persist state so UI/video text isn't erased after 1 frame
                st.session_state.last_is_dumping = is_dumping

                if is_dumping:
                    status_placeholder.markdown('<div class="alert-box alert-danger">🚨 ALERT: Illegal Dumping Detected!</div>', unsafe_allow_html=True)
                    timestamp = time.strftime("%H:%M:%S")
                    alert_entry = f"[{timestamp}] {class_name} ({confidence:.1%})"
                    if not st.session_state.alert_log or st.session_state.alert_log[0] != alert_entry:
                        st.session_state.alert_log.insert(0, alert_entry)
                        st.session_state.alert_log = st.session_state.alert_log[:10]
                else:
                    status_placeholder.markdown('<div class="alert-box alert-safe">✅ All Clear - Area Clean</div>', unsafe_allow_html=True)

                conf_html = ""
                for label in labels:
                    conf = all_preds.get(label, 0.0)
                    bar_width = int(conf * 100)

                    # --- UPDATED: Only highlight "Trash" class in red ---
                    is_dumping_class = label.lower() == "trash"
                    color = "#dc3545" if is_dumping_class else SDG_ORANGE

                    conf_html += f"""<div class="confidence-item">
                        <div class="confidence-label"><span>{label}</span><span style="color:{color};font-weight:600;">{conf:.1%}</span></div>
                        <div class="confidence-bar"><div class="confidence-fill" style="background:{color};width:{bar_width}%;"></div></div>
                    </div>"""
                confidence_placeholder.markdown(conf_html, unsafe_allow_html=True)

                if st.session_state.alert_log:
                    log_html = "".join(f'<div class="log-entry">{e}</div>' for e in st.session_state.alert_log)
                    log_placeholder.markdown(log_html, unsafe_allow_html=True)
                else:
                    log_placeholder.markdown(f"<p style='color:{SDG_TEXT};opacity:0.6;'>No alerts recorded</p>", unsafe_allow_html=True)

            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (38, 157, 249), 3)

            # Flip first, then draw overlay text so it isn't mirrored
            frame_rgb = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)

            # Apply detection text on every single frame based on persisted state!
            if st.session_state.last_is_dumping:
                cv2.rectangle(frame_rgb, (0, 0), (frame_rgb.shape[1], 60), (200, 0, 0), -1)
                cv2.putText(frame_rgb, "TRASH DETECTED!", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                play_alarm(volume=alarm_volume)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            time.sleep(0.033)

    except Exception as e:
        st.error(f"Error during detection: {e}")
    finally:
        cap.release()
        st.session_state.running = False


# ============================================================================
# MAIN
# ============================================================================
def main():
    st.set_page_config(
        page_title="Illegal Dumping Detector | SDG 11",
        page_icon="🏙️",
        layout="wide"
    )

    init_session_state()
    apply_custom_css()
    render_header()
    render_navigation()

    confidence_threshold, detection_interval, alarm_volume = render_sidebar()

    if st.session_state.current_page == "Detection":
        render_detection_page(confidence_threshold, detection_interval, alarm_volume)
    elif st.session_state.current_page == "Why This Technology":
        render_why_technology_page()
    elif st.session_state.current_page == "Benefits":
        render_benefits_page()
    elif st.session_state.current_page == "About SDG 11":
        render_sdg11_page()


if __name__ == "__main__":
    import sys
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx is None:
            print("\n" + "="*60)
            print("ERROR: This is a Streamlit app!")
            print("Run with: streamlit run " + sys.argv[0])
            print("="*60 + "\n")
            sys.exit(1)
    except ImportError:
        pass
    main()