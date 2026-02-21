import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# --- 1. PAGE SETUP & THEME ---
st.set_page_config(page_title="Universal AI Engine", layout="wide")

# Theme Toggle in Top Right
h_left, h_right = st.columns([6, 1])
with h_right:
    mode = st.toggle("🌙 Night Mode", value=False)

# Dynamic Theme Colors
if mode:
    bg_overlay, text_col, card_bg = "rgba(15, 23, 42, 0.9)", "#F8FAFC", "rgba(30, 41, 59, 0.95)"
else:
    bg_overlay, text_col, card_bg = "rgba(255, 255, 255, 0.1)", "#1E293B", "rgba(255, 255, 255, 0.85)"

# --- 2. CUSTOM CSS ---
st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient({bg_overlay}, {bg_overlay}), 
        url("https://images.unsplash.com/photo-1570129477492-45c003edd2be?auto=format&fit=crop&w=1920&q=80");
        background-size: cover; background-attachment: fixed;
        color: {text_col}; font-weight: 800 !important;
    }}
    .main-title {{ color: #7DD3FC; font-size: 50px; font-weight: 900; text-align: center; margin-top: -20px; }}
    
    /* Light Blue Button */
    div.stButton > button:first-child {{
        background-color: #7DD3FC !important; color: #0369A1 !important;
        font-weight: 900 !important; border-radius: 12px; height: 3.5em; width: 100%; border: none;
    }}
    
    /* Result Card Styling */
    .result-card {{ 
        background-color: {card_bg}; padding: 35px; border-radius: 20px; 
        border: 4px solid #7DD3FC; text-align: center; backdrop-filter: blur(15px);
        box-shadow: 0px 10px 30px rgba(0,0,0,0.3);
    }}

    /* Making all labels and text BOLD */
    label p, .stSlider p {{ font-weight: 900 !important; font-size: 18px !important; color: {text_col} !important; }}
    
    /* Toggle Color Fix */
    .st-emotion-cache-1dj0h3l {{ background-color: #7DD3FC !important; }}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Universal Neural Intelligence</h1>", unsafe_allow_html=True)
