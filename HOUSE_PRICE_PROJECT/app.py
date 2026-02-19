import streamlit as st
import numpy as np
import xgboost as xgb
import os

st.set_page_config(page_title="AI House Valuer", page_icon="🏠")

def load_model():
    # This checks the main folder AND subfolders
    possible_paths = [
        'house_price_model.json',
        'HOUSE_PRICE_PROJECT/house_price_model.json'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            model = xgb.Booster()
            model.load_model(path)
            return model
    return None

st.title("🏠 Luxury House Price Predictor")
st.write("Enter details below and click Predict.")

quality = st.slider("Overall Quality (1-10)", 1, 10, 6)
year = st.slider("Year Built", 1880, 2010, 1995)
area = st.number_input("Living Area (SqFt)", 500, 5000, 1500)

if st.button("Predict Price"):
    bst = load_model()
    if bst:
        # Quality, Area, Year
        features = np.array([[quality, area, year]])
        dmat = xgb.DMatrix(features)
        prediction = bst.predict(dmat)
        st.balloons()
        st.success(f"### Estimated Price: ${prediction[0]:,.2f}")
    else:
        st.error("Model file not found. Check if house_price_model.json is on GitHub main page.")
