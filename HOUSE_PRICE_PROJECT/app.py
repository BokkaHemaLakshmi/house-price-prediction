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

# The 3 inputs the user cares about
quality = st.slider("Overall Quality (1-10)", 1, 10, 6)
year = st.slider("Year Built", 1880, 2010, 1995)
area = st.number_input("Living Area (SqFt)", 500, 5000, 1500)

if st.button("Predict Price"):
    bst = load_model()
    if bst:
        try:
            # 1. List of ALL 75 features the model expects (from your error message)
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

            # 2. Create a dictionary with all 0s
            input_dict = {feat: [0] for feat in all_features}

            # 3. Fill in the actual user values
            input_dict['OverallQual'] = [quality]
            input_dict['GrLivArea'] = [area]
            input_dict['YearBuilt'] = [year]

            # 4. Convert to DataFrame and predict
            input_df = pd.DataFrame(input_dict)
            
            # Reorder columns to match exactly what the model expects
            input_df = input_df[all_features]
            
            dmat = xgb.DMatrix(input_df)
            prediction = bst.predict(dmat)
            dmat = xgb.DMatrix(input_df)
            prediction = bst.predict(dmat)
            
            # Professional Output
            st.markdown("---") # Adds a divider line
            st.subheader("Market Analysis Results")
            st.metric(label="Estimated Property Value", value=f"${prediction[0]:,.2f}")
            st.caption("Disclaimer: This AI-generated estimate is based on historical market data.")
            
            st.success(f"### Estimated Price: ${prediction[0]:,.2f}")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Model file not found.")
