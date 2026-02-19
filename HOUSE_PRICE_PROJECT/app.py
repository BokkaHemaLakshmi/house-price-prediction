
import streamlit as st
import numpy as np
import xgboost as xgb

st.set_page_config(page_title="AI House Valuer", page_icon="🏠")

# Load the AI model
def load_model():
    model = xgb.Booster()
    model.load_model('house_price_model.json')
    return model

st.title("🏠 Luxury House Price Predictor")
st.write("Enter the property details to get an instant AI valuation.")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    quality = st.slider("Overall Quality (1-10)", 1, 10, 6)
    year = st.slider("Year Built", 1880, 2010, 1995)
with col2:
    area = st.number_input("Living Area (SqFt)", 500, 5000, 1500)

# Prediction Button
if st.button("Predict Price"):
    try:
        bst = load_model()
        # Ensure the features match the 3 we trained on: Quality, Area, Year
        features = np.array([[quality, area, year]])
        dmat = xgb.DMatrix(features)
        prediction = bst.predict(dmat)
        
        st.balloons()
        st.success(f"### Estimated Price: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error("Error: Make sure 'house_price_model.json' is uploaded to GitHub.")
