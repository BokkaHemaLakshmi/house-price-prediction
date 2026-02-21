import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Neural Estate Intelligence", layout="wide", page_icon="🏠")

# --- 2. SIMPLE & CLEAN PROFESSIONAL STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #1E293B; }
    .main-header { color: #1E3A8A; font-size: 38px; font-weight: 800; text-align: center; margin-bottom: 10px; }
    .result-card { 
        background-color: #F8FAFC; 
        border: 2px solid #2563EB; 
        padding: 25px; 
        border-radius: 12px; 
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .feature-box {
        padding: 15px;
        background-color: #F1F5F9;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
    }
    div.stButton > button:first-child {
        background-color: #2563EB;
        color: white;
        width: 100%;
        border-radius: 8px;
        height: 3em;
    }
    /* Hide Streamlit elements for a clean look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('house_model.joblib')
    except:
        return None

model = load_model()

# --- 4. DATA STRUCTURE (75 FEATURES) ---
# Replace these with your actual 75 feature
