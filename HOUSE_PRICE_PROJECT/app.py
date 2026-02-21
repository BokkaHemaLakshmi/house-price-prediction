import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBRegressor

# --- 1. THEME & HEADER ---
st.set_page_config(page_title="Universal AI Engine", layout="wide")
h_left, h_right = st.columns([6, 1])
with h_right:
    mode = st.toggle("🌙 Night Mode", value=False)

if mode:
    bg_overlay, text_col, card_bg, grid_col = "rgba(15, 23, 42, 0.9)", "#F8FAFC", "rgba(30, 41, 59, 0.95)", "#334155"
else:
    bg_overlay, text_col, card_bg, grid_col = "rgba(255, 255, 255, 0.1)", "#1E293B", "rgba(255, 255, 255, 0.85)", "#E2E8F0"

# --- 2. SKY BLUE CSS ---
st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient({bg_overlay}, {bg_overlay}), 
        url("https://images.unsplash.com/photo-1570129477492-45c003edd2be?auto=format&fit=crop&w=1920&q=80");
        background-size: cover; background-attachment: fixed;
        color: {text_col}; font-weight: 800 !important;
    }}
    .main-title {{ color: #7DD3FC; font-size: 50px; font-weight: 900; text-align: center; }}
    div.stButton > button:first-child {{
        background-color: #7DD3FC !important; color: #0369A1 !important;
        font-weight: 900 !important; border-radius: 12px; height: 3.5em; width: 100%; border: none;
    }}
    .result-card {{ 
        background-color: {card_bg}; padding: 35px; border-radius: 20px; 
        border: 4px solid #7DD3FC; text-align: center; backdrop-filter: blur(15px);
    }}
    .st-emotion-cache-1dj0h3l {{ background-color: #7DD3FC !important; }}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Neural Estate Intelligence</h1>", unsafe_allow_html=True)

# --- 3. DATA UPLOAD SECTION ---
st.markdown("### 📥 1. Upload Dataset")
file = st.file_uploader("Upload Ames train.csv", type="csv")

if file:
    df = pd.read_csv(file).dropna()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # --- 4. DATA VISUALIZATION (THE GRAPH) ---
    st.divider()
    st.markdown("### 📊 Market Distribution")
    # Show a graph of the target price automatically
    target_col = "SalePrice" if "SalePrice" in num_cols else num_cols[-1]
    fig = px.histogram(df, x=target_col, title=f"Property Value Distribution in {file.name}", color_discrete_sequence=['#7DD3FC'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=text_col)
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. DYNAMIC FEATURE SLIDERS ---
    st.divider()
    st.markdown("### 🛠️ 2. Set Parameters & Predict")
    features = st.multiselect("Select features to use", [c for c in num_cols if c != target_col], default=num_cols[:4])
    
    if features:
        model = XGBRegressor().fit(df[features], df[target_col])
        
        col_in, col_res = st.columns([3, 2], gap="large")
        with col_in:
            inputs = {}
            for f in features:
                inputs[f] = st.slider(f, float(df[f].min()), float(df[f].max()), float(df[f].mean()))
        
        with col_res:
            if st.button("🚀 PREDICT PRICE"):
                pred = model.predict(pd.DataFrame([inputs]))[0]
                st.markdown(f"""
                    <div class='result-card'>
                        <h2 style='color: #0369A1;'>ESTIMATED VALUE</h2>
                        <h1 style='color: #7DD3FC; font-size: 55px;'>${pred:,.2f}</h1>
                    </div>
                """, unsafe_allow_html=True)
else:
    # This shows when the page is empty
    st.info("Please upload a CSV file to see the Graphs, Sliders, and Prediction tools!")
