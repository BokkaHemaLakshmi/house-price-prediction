import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Neural Estate Intelligence", layout="wide")

# --- 2. HEADER & MODE TOGGLE ---
# Using columns to push the toggle to the top right
h_left, h_right = st.columns([5, 1])

with h_right:
    # This is the toggle you wanted to change from red to light blue
    mode = st.toggle("🌙 Night Mode", value=False)

# Theme Logic
if mode:
    bg_overlay = "rgba(15, 23, 42, 0.85)"
    text_color = "#F8FAFC"
    card_bg = "rgba(30, 41, 59, 0.9)"
    accent = "#7DD3FC"  # Sky Blue
    grid_color = "#334155"
else:
    bg_overlay = "rgba(255, 255, 255, 0.1)" # Fully clear background
    text_color = "#1E293B"
    card_bg = "rgba(255, 255, 255, 0.8)"
    accent = "#0369A1" # Deep Blue for readability
    grid_color = "#E2E8F0"

# --- 3. CUSTOM CSS ---
st.markdown(f"""
    <style>
    /* Global Styles */
    .stApp {{
        background: linear-gradient({bg_overlay}, {bg_overlay}), 
        url("https://images.unsplash.com/photo-1570129477492-45c003edd2be?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-attachment: fixed;
        color: {text_color};
        font-weight: 700 !important; /* Bold font for everything */
    }}

    /* Sky Blue Heading */
    .main-title {{
        color: #7DD3FC;
        font-size: 50px;
        font-weight: 900;
        text-align: center;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.4);
        margin-top: -20px;
    }}

    /* Light Blue Button & Toggle Fix */
    div.stButton > button:first-child {{
        background-color: #7DD3FC !important;
        color: #0369A1 !important;
        border: none;
        font-weight: 800 !important;
        border-radius: 12px;
        height: 3.5em;
        width: 100%;
    }}

    /* CSS to force the toggle color to Light Blue instead of red */
    .st-emotion-cache-1dj0h3l {{ background-color: #7DD3FC !important; }}

    .result-card {{ 
        background-color: {card_bg}; 
        padding: 30px; 
        border-radius: 20px; 
        border: 3px solid #7DD3FC;
        text-align: center;
        backdrop-filter: blur(10px);
    }}

    /*
