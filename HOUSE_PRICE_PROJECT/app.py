import streamlit as st
import numpy as np
import xgboost as xgb
import os
import pandas as pd

# 1. PAGE CONFIG (Must be at the top)
st.set_page_config(page_title="AI House Valuer", page_icon="🏠")

# 2. THE CLEANUP SUIT (Hides Streamlit branding for Lovable)
hide_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {padding-top: 2rem;}
            </style>
            """
st.markdown(hide_style, unsafe_allow_html=True)

# 3. MODEL LOADING FUNCTION
def load_model():
    possible_paths = ['house_price_model.json', 'HOUSE_PRICE_PROJECT/house_price_model.json']
    for path in possible_paths:
        if os.path.exists(path):
            model = xgb.Booster()
            model.load_model(path)
            return model
    return None

# 4. USER INTERFACE
st.title("🏠 Luxury House Price Predictor")
st.write("Enter the property details below to see the AI-estimated market value.")

# Create two columns for a cleaner look
col1, col2 = st.columns(2)

with col1:
    quality = st.slider("Overall Quality (1-10)", 1, 10, 6)
    year = st.slider("Year Built", 1880, 2024, 1995)

with col2:
    area = st.number_input("Living Area (SqFt)", 500, 10000, 1500)
    rooms = st.slider("Total Rooms", 2, 15, 6)

# 5. PREDICTION LOGIC
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

            # Create baseline dictionary (Filling with average values to avoid $11 error)
            input_dict = {feat: [0] for feat in all_features}
            input_dict['MSSubClass'] = [60]
            input_dict['LotArea'] = [int(area * 6)] 
            input_dict['OverallCond'] = [5]
            input_dict['FullBath'] = [2]
            input_dict['BedroomAbvGr'] = [3]
            input_dict['GarageCars'] = [2]
            
            # Applying User Inputs
            input_dict['OverallQual'] = [quality]
            input_dict['GrLivArea'] = [area]
            input_dict['YearBuilt'] = [year]
            input_dict['YearRemodAdd'] = [year]
            input_dict['TotRmsAbvGrd'] = [rooms]

            # Prepare data for model
            input_df = pd.DataFrame(input_dict)
            input_df = input_df[all_features]
            dmat = xgb.DMatrix(input_df)
            
            # Predict
            prediction = bst.predict(dmat)[0]
            
            # Log-scale Correction (Ensures price is in $ and not a small log number)
            if prediction < 25:
                prediction = np.exp(prediction)

            # SHOW RESULTS
            st.markdown("---")
            st.subheader("Valuation Summary")
            st.metric(label="Estimated Property Value", value=f"${prediction:,.2f}")
            st.info("AI Analysis complete. This estimate is based on the 75-feature XGBoost model.")

        except Exception as e:
            st.error(f"Error during calculation: {e}")
    else:
        st.error("Model file not found. Please check your GitHub folder for 'house_price_model.json'.")
