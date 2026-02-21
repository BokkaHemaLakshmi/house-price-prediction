import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Neural Estate Intelligence", layout="wide", page_icon="🏠")

# --- 2. THEME & BACKGROUND LOGIC (TOP RIGHT TOGGLE) ---
# Create a header row to place toggle on the right
header_left, header_right = st.columns([5, 1])

with header_right:
    # STEP 2: Icon-style toggle for Night Mode
    mode = st.toggle("🌙 Night Mode", value=False)

# Define Colors based on Theme
if mode:
    bg_overlay = "rgba(15, 23, 42, 0.85)"  # Dark Slate
    text_color = "#F8FAFC"
    card_color = "rgba(30, 41, 59, 0.9)"
    btn_color = "#3B82F6"
else:
    bg_overlay = "rgba(255, 255, 255, 0.85)"  # Pure White
    text_color = "#1E293B"
    card_color = "rgba(248, 250, 252, 0.9)"
    btn_color = "#2563EB"

# STEP 3: Apply Background Image (Ames Style) and Theme Colors
st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient({bg_overlay}, {bg_overlay}), 
        url("https://images.unsplash.com/photo-1570129477492-45c003edd2be?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-
