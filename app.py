from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Path is relative to this file so the app runs from any clone location
MODEL_PATH = Path(__file__).resolve().parent / "house_price_production_model.pkl"

# Loads the model
model_artifact = joblib.load(MODEL_PATH)
pipeline = model_artifact["pipeline"]

# City/zip options come straight from the fitted encoder, so the dropdowns
# always match what the model was actually trained on.
ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
city_options = sorted(ohe.categories_[0])
statezip_options = sorted(ohe.categories_[1])

st.title("🏠 House Price Prediction")
st.write("Enter the property's details below:")

# Inputs required fields
bedrooms = st.number_input("Bedrooms", min_value=0.0, value=3.0, step=1.0)
bathrooms = st.number_input("Bathrooms", min_value=0.0, value=2.0, step=0.25)
sqft_living = st.number_input("Living Area (sqft)", min_value=0, value=1800)
sqft_lot = st.number_input("Lot Size (sqft)", min_value=0, value=7500)
floors = st.number_input("Floors", min_value=1.0, value=1.0, step=0.5)
waterfront = st.selectbox("Waterfront", ["No", "Yes"])
view = st.slider("View Score (0 = none, 4 = excellent)", 0, 4, 0)
condition = st.slider("Condition (1 = poor, 5 = excellent)", 1, 5, 3)
sqft_above = st.number_input("Above-Ground Area (sqft)", min_value=0, value=1500)
sqft_basement = st.number_input("Basement Area (sqft)", min_value=0, value=300)
yr_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=1985)
yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0, max_value=2026, value=0)
month = st.slider("Month of Sale", 1, 12, 6)
city = st.selectbox("City", city_options, index=city_options.index("Seattle") if "Seattle" in city_options else 0)
statezip = st.selectbox("State + ZIP", statezip_options)

if st.button("Predict"):
    data = {
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "sqft_living": [sqft_living],
        "sqft_lot": [sqft_lot],
        "floors": [floors],
        "waterfront": [1 if waterfront == "Yes" else 0],
        "view": [view],
        "condition": [condition],
        "sqft_above": [sqft_above],
        "sqft_basement": [sqft_basement],
        "yr_built": [yr_built],
        "yr_renovated": [yr_renovated],
        "month": [month],
        "city": [city],
        "statezip": [statezip],
    }
    df = pd.DataFrame(data)
    predicted_price = pipeline.predict(df)[0]
    st.subheader(f"Estimated Price: ${predicted_price:,.0f}")
    st.caption(
        f"Model: {model_artifact['model_name']} · trained on Seattle-area listings, "
        "May-Jul 2014 — treat this as a ballpark, not an appraisal."
    )
