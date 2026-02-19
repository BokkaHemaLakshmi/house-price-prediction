import streamlit as st
import numpy as np
import xgboost as xgb
import os
import pandas as pd

st.set_page_config(page_title="AI House Valuer", page_icon="🏠")

def load_model():
    possible_paths = ['house_price_model.json', 'HOUSE_PRICE_PROJECT/house_price_model.json']
    for path in possible_paths:
        if os.path.exists(path):
            model = xgb.Booster()
            model.load_model(path)
            return model
    return None

st.title("🏠 Luxury House Price Predictor")

quality = st.slider("Overall Quality (1-10)", 1, 10, 6)
year = st.slider("Year Built", 1880, 2010, 1995)
area = st.number_input("Living Area (SqFt)", 500, 5000, 1500)

if st.button("Predict Price"):
    bst = load_model()
    if bst:
        try:
            # We create a DataFrame so we can give names to the columns
            # IMPORTANT: The names 'OverallQual', 'GrLivArea', 'YearBuilt' 
            # must match exactly what you used during training!
            input_data = pd.DataFrame({
                'OverallQual': [quality],
                'GrLivArea': [area],
                'YearBuilt': [year]
            })
            
            # Convert to DMatrix with feature names
            dmat = xgb.DMatrix(input_data)
            prediction = bst.predict(dmat)
            
            st.balloons()
            st.success(f"### Estimated Price: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Feature Error: {e}")
            st.info("Try changing the column names in the code to match your training data.")
    else:
        st.error("Model file not found.")
