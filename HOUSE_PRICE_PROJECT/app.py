import streamlit as st
import numpy as np
import xgboost as xgb
import os
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="AI House Valuer", page_icon="🏠")

# 2. Professional UI "Suit" (Hides Streamlit decorations)
hide_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {padding-top: 1rem;}
            </style>
            """
st.markdown(hide_style, unsafe_allow_html=True)

# 3. Model Loading Logic
def load_model():
    # Checking both possible paths
    possible_paths = ['house_price_model.json', 'HOUSE_PRICE_PROJECT/house_price_model.json']
    for path in possible_paths:
        if os.path.exists(path):
            model = xgb.Booster()
            model.load_model(path)
            return model
    return None

# 4. App Interface
st.title("🏠 Luxury House Price Predictor")
st.write("Enter property details to get an AI-powered market valuation.")

# User Inputs organized in columns
col1, col2 = st.columns(2)
with col1:
    quality = st.slider("Overall Quality (1-10)", 1, 10, 6)
    year = st.slider("Year Built", 1880, 2024, 1995)
with col2:
    area = st.number_input("Living Area (SqFt)", 500, 10000, 1500)
    rooms = st.slider("Total Rooms", 2, 15, 6)

if st.button("Calculate Market Value"):
    bst = load_model()
    if bst:
        try:
            # All 75 features the model expects
            all_features = [
                'Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'LotShape', 'LandContour', 
                'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 
                'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 
                'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 
                'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', 
                '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
                'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 
                'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 
                'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 
                'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'
            ]

            # Create baseline dictionary with "Standard House" values
            input_dict = {feat: [0] for feat in all_features}
            
            # Setting default averages so the model recognizes it as a real house
            input_dict['MSSubClass'] = [60]
            input_dict['LotArea'] = [int(area * 6)] 
            input_dict['OverallCond'] = [5]
            input_dict['FullBath'] = [2]
            input_dict['BedroomAbvGr'] = [3]
            input_dict['GarageCars'] = [2]
            
            # Applying User Inputs from Sliders
