# =============================================================================
# Seed Classification Dashboard
# -----------------------------------------------------------------------------
# Streamlit application for classifying seeds using a machine learning
# pipeline.
# =============================================================================

import streamlit as st
import pandas as pd
import pickle
import os
import base64
import streamlit.components.v1 as components
from sklearn.base import BaseEstimator, TransformerMixin


# =============================================================================
# Custom Classes for Pickle Loading
# =============================================================================
# Pickle requires custom classes used during training to be present in the
# loading environment. This defines the structure so pickle can load the model.
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # If your original OutlierCapper performed specific capping math
        # during inference, replace this passthrough with your original logic.
        return X


# =============================================================================
# Streamlit Application Configuration
# =============================================================================

st.set_page_config(
    page_title="Dry Bean Morphological Classifier", page_icon="🌱", layout="centered"
)

# =============================================================================
# Custom UI Styling (Preserved)
# =============================================================================

st.markdown(
    """
    <style>
    
    @import url('https://api.fontshare.com/v2/css?f[]=clash-display@600,700,800&f[]=satoshi@400,500,700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@200;300;400;500;600;700;800&display=swap');

    
    html, body, [class*="css"], .stMarkdown, p, span, div, li {
        font-family: 'Satoshi', sans-serif !important;
    }

    
    h1, h2, h3, .gradient-text, .section-header {
        font-family: 'Clash Display', sans-serif !important;
    }
    
    button, .stButton > button,
    label, .stLabel,
    input, select, textarea,
    div[data-baseweb="slider"] *,
    div[data-baseweb="select"] * {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: 0.03em !important; 
    }

    
    
    div[data-testid="stSlider"], div[data-testid="stSelectbox"], div[data-testid="stNumberInput"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 8px 0px !important;
        margin-bottom: 24px !important;
    }
    
    
    div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label {
        color: #B0B0B0 !important; 
        font-size: 0.85rem !important;
        font-weight: 600 !important; 
        letter-spacing: 0.1em !important;
        text-transform: uppercase;
        margin-bottom: 12px !important;
        transition: color 0.4s ease !important;
    }
    div[data-testid="stSlider"]:hover label, div[data-testid="stSelectbox"]:hover label, div[data-testid="stNumberInput"]:hover label {
        color: #FFFFFF !important; 
    }

    
    /* Aggressively target all layers of the input component to be fully transparent */
    div[data-baseweb="select"], 
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"], 
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] button, 
    div[data-baseweb="input"] input,
    div[data-testid="stNumberInputStepUp"],
    div[data-testid="stNumberInputStepDown"],
    div[data-testid="stNumberInput"] button {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    /* Apply the light grey border ONLY to the main wrappers */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        border: 1px solid transparent !important;
        border-radius: 6px !important;
        padding: 0 4px !important;
        box-shadow: none !important; 
    }

    /* Hover and focus states for the border */
    div[data-baseweb="select"] > div:hover, div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:hover, div[data-baseweb="input"] > div:focus-within {
        box-shadow: none !important;
        border: 1px solid transparent !important;
    }
    
    /* Font colors */
    div[data-baseweb="select"] *, div[data-baseweb="input"] * {
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        color: #CCCCCC !important; 
        transition: color 0.3s ease !important;
    }
    
    div[data-baseweb="select"]:hover *, div[data-baseweb="input"]:hover *,
    div[data-baseweb="select"]:focus-within *, div[data-baseweb="input"]:focus-within * {
        color: #FACC15 !important; 
    }

    /* Target specifically the +/- button text/icons to glow yellow on hover */
    [data-testid="stNumberInputStepUp"]:hover,
    [data-testid="stNumberInputStepDown"]:hover,
    [data-testid="stNumberInputStepUp"]:hover *,
    [data-testid="stNumberInputStepDown"]:hover *,
    div[data-baseweb="input"] button:hover,
    div[data-baseweb="input"] button:hover * {
        color: #FACC15 !important;
        fill: #FACC15 !important;
        stroke: #FACC15 !important;
    }
    
    .section-header {
        color: #A0A0A0;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-top: 50px;
        margin-bottom: 25px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05); 
        background: none;
        box-shadow: none;
        display: block;
    }
    
    .element-container:has(.stButton) {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        margin-top: 18px !important;
    }

    div.stButton {
        width: auto !important;
        display: flex !important;
        justify-content: center !important;
    }

    div.stButton > button {
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: auto !important;
        min-width: 0 !important;
        padding: 13px 30px !important;
        border-radius: 8px !important;
        background: rgba(255,255,255,0.045) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #D8D8D8 !important;
        backdrop-filter: blur(10px) !important;
        overflow: hidden !important;

        transition:
            transform 0.45s cubic-bezier(0.16, 1, 0.3, 1),
            background 0.45s ease,
            border-color 0.45s ease,
            box-shadow 0.45s ease !important;

        box-shadow:
            0 0 0 rgba(250, 204, 21, 0),
            0 8px 30px rgba(0,0,0,0.18);
    }

    div.stButton > button p,
    div.stButton > button span {
        margin: 0 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 11px !important;
        font-weight: 800 !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        transition: color 0.35s ease !important;
    }

    div.stButton > button::before {
        content: "";
        position: absolute;
        inset: -40%;
        background:
            radial-gradient(
                circle at center,
                rgba(250, 204, 21, 0.18) 0%,
                rgba(250, 204, 21, 0.08) 30%,
                transparent 70%
            );

        opacity: 0;

        transform: translateX(-30%) translateY(10%) scale(0.8);

        transition:
            opacity 0.6s ease,
            transform 0.8s cubic-bezier(0.16,1,0.3,1);

        pointer-events: none;
    }

    div.stButton > button:hover {
        transform:
            translateY(-2px)
            scale(1.015);

        background: rgba(255,255,255,0.065) !important;

        border-color: rgba(250, 204, 21, 0.22) !important;

        box-shadow:
            0 0 40px rgba(250, 204, 21, 0.10),
            0 12px 40px rgba(0,0,0,0.24);
    }

    div.stButton > button:hover::before {
        opacity: 1;

        transform:
            translateX(15%)
            translateY(-10%)
            scale(1.15);
    }

    div.stButton > button:hover p,
    div.stButton > button:hover span {
        color: white !important;
    }

    div.stButton > button:active {
        transform:
            translateY(0px)
            scale(0.985);
    }

    @keyframes textShine {
        0% { background-position: 0% center; }
        100% { background-position: 100% center; }
    }
    .gradient-text {
        background: linear-gradient(120deg, #FACC15 30%, #FEF08A 50%, #FACC15 70%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: textShine 4s cubic-bezier(0.4, 0, 0.2, 1) infinite alternate;
    }
    
    body::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.018;
        z-index: 0;
        background-image:
            radial-gradient(rgba(255,255,255,0.14) 0.5px, transparent 0.5px);
        background-size: 4px 4px;
    }

    .main .block-container {
        animation: pageReveal 900ms cubic-bezier(0.16, 1, 0.3, 1);
    }

    @keyframes pageReveal {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .result-reveal {
        animation: resultReveal 650ms cubic-bezier(0.16, 1, 0.3, 1);
    }

    @keyframes resultReveal {
        from {
            opacity: 0;
            transform: translateY(14px);
            filter: blur(6px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
            filter: blur(0px);
        }
    }
    </style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Model Loading
# =============================================================================


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


try:
    pipeline = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file 'model.pkl' not found.")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# Hero Section
# =============================================================================

st.markdown(
    """
    <h1 style='font-size: 60px; margin-bottom: 0; line-height: 1.1; display: flex; flex-direction: column; align-items: center; text-align: center;'>
        <span>Dry Bean Morphological</span>
        <span class='gradient-text' style='font-size: 68px; font-weight: 800;'>Classifier</span>
    </h1>
    <p style='color: #888; font-size: 16px; margin-top: 20px; text-align: center; font-weight: 500;'>
        Precision seed classification powered by geometric analysis.
    </p>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

# =============================================================================
# User Input Section (Categorized)
# =============================================================================

st.markdown(
    "<div class='section-header'>📏 Size Dimensions</div>", unsafe_allow_html=True
)
col1, col2 = st.columns(2)
with col1:
    area = st.number_input(
        "Area", min_value=0.0, max_value=500000.0, value=28395.0, format="%.5f"
    )
    minor_axis = st.number_input(
        "MinorAxisLength",
        min_value=0.0,
        max_value=2000.0,
        value=173.88875,
        format="%.5f",
    )
with col2:
    perimeter = st.number_input(
        "Perimeter", min_value=0.0, max_value=3000.0, value=610.29100, format="%.5f"
    )


st.markdown(
    "<div class='section-header'>💠 Shape Characteristics</div>", unsafe_allow_html=True
)
col3, col4 = st.columns(2)
with col3:
    solidity = st.number_input(
        "Solidity", min_value=0.0, max_value=1.0, value=0.98886, format="%.5f"
    )
    roundness = st.number_input(
        "roundness", min_value=0.0, max_value=1.0, value=0.95803, format="%.5f"
    )
with col4:
    extent = st.number_input(
        "Extent", min_value=0.0, max_value=1.0, value=0.76392, format="%.5f"
    )
    compactness = st.number_input(
        "Compactness", min_value=0.0, max_value=1.0, value=0.91336, format="%.5f"
    )


st.markdown(
    "<div class='section-header'>📐 Shape Factors</div>", unsafe_allow_html=True
)
col5, col6 = st.columns(2)
with col5:
    shape_factor1 = st.number_input(
        "ShapeFactor1", min_value=0.0, max_value=1.0, value=0.00733, format="%.5f"
    )
    shape_factor4 = st.number_input(
        "ShapeFactor4", min_value=0.0, max_value=1.0, value=0.99872, format="%.5f"
    )
with col6:
    shape_factor2 = st.number_input(
        "ShapeFactor2", min_value=0.0, max_value=1.0, value=0.00315, format="%.5f"
    )

# =============================================================================
# Prediction Execution
# =============================================================================

if st.button("Run Classification"):
    input_data = pd.DataFrame(
        {
            "Area": [area],
            "Perimeter": [perimeter],
            "MinorAxisLength": [minor_axis],
            "Solidity": [solidity],
            "ShapeFactor4": [shape_factor4],
            "Extent": [extent],
            "roundness": [roundness],
            "Compactness": [compactness],
            "ShapeFactor1": [shape_factor1],
            "ShapeFactor2": [shape_factor2],
        }
    )

    with st.spinner("Processing through pipeline..."):
        try:
            prediction = pipeline.predict(input_data)

            st.markdown(
                """
                <div class="result-reveal" style="text-align:center; margin-top:38px; padding:42px 20px 12px;">
                    <p style="color:#777; font-size:11px; font-weight:700; letter-spacing:0.22em; text-transform:uppercase; margin:0 0 18px;">
                        Prediction Result
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<h1 style='text-align:center; font-size:48px; margin:0; font-family:Clash Display, sans-serif; line-height:1;' class='gradient-text'>{prediction[0]}</h1>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div style='width:46px; height:1px; background:rgba(250, 204, 21, 0.22); margin:28px auto 0;'></div>",
                unsafe_allow_html=True,
            )

            components.html(
                """
                <script>
                    const result = window.parent.document.querySelector('.result-reveal');

                    if (result) {
                        result.scrollIntoView({
                            behavior: 'smooth'
                        });
                    }
                </script>
                """,
                height=0,
            )

        except Exception as e:
            st.error(f"Computation Error: {e}")

st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)

# =============================================================================
# Frontend Interaction Enhancements (Preserved)
# =============================================================================

components.html(
    """
    <script>
    const doc = window.parent.document;

    function centerButton() {
        doc.querySelectorAll('.stButton').forEach(el => {
            el.style.setProperty('display', 'flex', 'important');
            el.style.setProperty('justify-content', 'center', 'important');
            let parent = el.parentElement;
            while (parent) {
                parent.style.setProperty('display', 'flex', 'important');
                parent.style.setProperty('justify-content', 'center', 'important');
                if (parent.classList.contains('block-container')) break;
                parent = parent.parentElement;
            }
        });
    }

    centerButton();
    new MutationObserver(centerButton).observe(doc.body, { childList: true, subtree: true });

    const buttons = doc.querySelectorAll('.stButton > button');

    buttons.forEach(btn => {

        btn.addEventListener('mousemove', e => {

            const rect = btn.getBoundingClientRect();

            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const moveX = (x - rect.width / 2) * 0.04;
            const moveY = (y - rect.height / 2) * 0.10;

            btn.style.transform = `
                translate(${moveX}px, ${moveY - 2}px)
                scale(1.015)
            `;
        });

        btn.addEventListener('mouseleave', () => {
            btn.style.transform = '';
        });
    });
    </script>
    """,
    height=0,
    width=0,
)
