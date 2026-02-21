import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# --- 1. PAGE SETUP & THEME ---
st.set_page_config(page_title="Universal Neural Engine", layout="wide")

# Header & Mode Toggle
h_left, h_right = st.columns([5, 1])
with h_right:
    mode = st.toggle("🌙 Night Mode", value=False)

if mode:
    bg_overlay, text_col, card_bg = "rgba(15, 23, 42, 0.85)", "#F8FAFC", "rgba(30, 41, 59, 0.9)"
    accent = "#7DD3FC"
else:
    bg_overlay, text_col, card_bg = "rgba(255, 255, 255, 0.1)", "#1E293B", "rgba(255, 255, 255, 0.8)"
    accent = "#0369A1"

# --- 2. CUSTOM CSS ---
st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient({bg_overlay}, {bg_overlay}), 
        url("https://images.unsplash.com/photo-1570129477492-45c003edd2be?auto=format&fit=crop&w=1920&q=80");
        background-size: cover; background-attachment: fixed;
        color: {text_col}; font-weight: 700 !important;
    }}
    .main-title {{ color: #7DD3FC; font-size: 42px; font-weight: 900; text-align: center; }}
    div.stButton > button:first-child {{
        background-color: #7DD3FC !important; color: #0369A1 !important;
        font-weight: 800 !important; border-radius: 12px; height: 3em; width: 100%;
    }}
    .result-card {{ 
        background-color: {card_bg}; padding: 25px; border-radius: 20px; 
        border: 3px solid #7DD3FC; text-align: center; backdrop-filter: blur(10px);
    }}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Universal Dynamic Predictor</h1>", unsafe_allow_html=True)

# --- 3. DYNAMIC FEATURE LOGIC ---
st.sidebar.header("📥 1. Upload Dataset")
file = st.sidebar.file_uploader("Upload any CSV file", type="csv")

if file:
    df = pd.read_csv(file).dropna()
    all_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    st.sidebar.subheader("⚙️ 2. Map Variables")
    target = st.sidebar.selectbox("Select Target (Value to Predict)", all_cols, index=len(all_cols)-1)
    features = st.sidebar.multiselect("Select Features (Inputs)", [c for c in all_cols if c != target], default=[c for c in all_cols if c != target][:4])

    if features:
        # Train Model Live
        model = XGBRegressor().fit(df[features], df[target])
        
        col_in, col_out = st.columns([3, 2], gap="large")
        
        with col_in:
            st.subheader("🛠️ Adjust Features")
            user_inputs = {}
            # DYNAMICALLY CREATE SLIDERS BASED ON CSV
            for f in features:
                f_min, f_max = float(df[f].min()), float(df[f].max())
                f_avg = float(df[f].mean())
                user_inputs[f] = st.slider(f"Adjust {f}", f_min, f_max, f_avg)

        with col_out:
            st.subheader("💰 Live Prediction")
            input_df = pd.DataFrame([user_inputs])
            prediction = model.predict(input_df)[0]
            
            st.markdown(f"""
                <div class='result-card'>
                    <p style='opacity: 0.8;'>PREDICTED {target.upper()}</p>
                    <h1 style='color: #0369A1; font-size: 45px;'>${prediction:,.2f}</h1>
                    <p style='color: #7DD3FC;'>Trained on {len(df)} records</p>
                </div>
            """, unsafe_allow_html=True)
else:
    st.info("Please upload a CSV file in the sidebar to start the Dynamic Engine. (Try uploading the Ames train.csv!)")

st.markdown("<footer style='text-align:center; margin-top:50px;'><hr>Built by Garu | Dynamic ML Framework</footer>", unsafe_allow_html=True)
