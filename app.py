import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Delhi Mumbai House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------------------------
# LOAD FILES
# ---------------------------------------------------
model = joblib.load("models/linear_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")
model_columns = joblib.load("models/model_columns.pkl")
city_locations = joblib.load("models/city_locations.pkl")

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("🏠 Delhi Mumbai House Price Predictor")

st.markdown("""
Predict intelligent house prices for **Delhi** and **Mumbai**
using Machine Learning.
""")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("🏡 Property Details")

# ---------------------------------------------------
# CITY
# ---------------------------------------------------
city = st.sidebar.selectbox(
    "Select City",
    ["Delhi", "Mumbai"]
)

# ---------------------------------------------------
# LOCATION
# ---------------------------------------------------
location = st.sidebar.selectbox(
    "Select Location",
    city_locations[city]
)

# ---------------------------------------------------
# AREA
# ---------------------------------------------------
area = st.sidebar.number_input(
    "Area (sqft)",
    min_value=200,
    max_value=10000,
    value=1000
)

# ---------------------------------------------------
# BEDROOMS
# ---------------------------------------------------
bedrooms = st.sidebar.slider(
    "Bedrooms",
    1,
    10,
    2
)

# ---------------------------------------------------
# AMENITIES
# ---------------------------------------------------
gym = st.sidebar.checkbox("Gymnasium")

pool = st.sidebar.checkbox("Swimming Pool")

parking = st.sidebar.checkbox("Car Parking")

lift = st.sidebar.checkbox("Lift Available")

security = st.sidebar.checkbox("24X7 Security")

ac = st.sidebar.checkbox("Air Conditioning")

# Convert True/False to 1/0
gym = int(gym)
pool = int(pool)
parking = int(parking)
lift = int(lift)
security = int(security)
ac = int(ac)

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------

# Luxury Score
luxury_score = (
    area * 0.35 +
    bedrooms * 0.25 +
    pool * 0.10 +
    gym * 0.10 +
    lift * 0.05 +
    ac * 0.05 +
    parking * 0.10
)

# Price per sqft placeholder
price_per_sqft = 0

# ---------------------------------------------------
# SIZE CATEGORY
# ---------------------------------------------------
if area < 800:
    size_category = "Small"

elif area < 1500:
    size_category = "Medium"

elif area < 2500:
    size_category = "Large"

else:
    size_category = "Luxury"

# ---------------------------------------------------
# LUXURY CATEGORY
# ---------------------------------------------------
if luxury_score < 800:
    luxury_category = "Basic"

elif luxury_score < 1500:
    luxury_category = "Premium"

else:
    luxury_category = "Luxury"

# ---------------------------------------------------
# INPUT DATAFRAME
# ---------------------------------------------------
input_data = pd.DataFrame({
    "Area": [area],
    "No_of_Bedrooms": [bedrooms],
    "Gymnasium": [gym],
    "SwimmingPool": [pool],
    "CarParking": [parking],
    "LiftAvailable": [lift],
    "24X7Security": [security],
    "AC": [ac],
    "Luxury_Score": [luxury_score],
    "Price_per_sqft": [price_per_sqft]
})

# ---------------------------------------------------
# CITY ENCODING
# ---------------------------------------------------
if "City_Mumbai" in model_columns:
    input_data["City_Mumbai"] = 1 if city == "Mumbai" else 0

# ---------------------------------------------------
# SIZE CATEGORY ENCODING
# ---------------------------------------------------
size_col = f"Size_category_{size_category}"

if size_col in model_columns:
    input_data[size_col] = 1

# ---------------------------------------------------
# LUXURY CATEGORY ENCODING
# ---------------------------------------------------
lux_col = f"Luxury_Category_{luxury_category}"

if lux_col in model_columns:
    input_data[lux_col] = 1

# ---------------------------------------------------
# LOCATION ENCODING
# ---------------------------------------------------
location_col = f"Location_{location}"

if location_col in model_columns:
    input_data[location_col] = 1

# ---------------------------------------------------
# ADD MISSING COLUMNS
# ---------------------------------------------------
for col in model_columns:

    if col not in input_data.columns:
        input_data[col] = 0

# ---------------------------------------------------
# COLUMN ORDER
# ---------------------------------------------------
input_data = input_data[model_columns]

# ---------------------------------------------------
# SCALE INPUT
# ---------------------------------------------------
input_scaled = scaler.transform(input_data)

# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------
if st.button("🔮 Predict Price"):

    prediction = model.predict(input_scaled)

    predicted_price = prediction[0]

    # ---------------------------------------------------
    # MAIN RESULT
    # ---------------------------------------------------
    st.success(
        f"🏠 Estimated Price: ₹ {predicted_price:,.0f}"
    )

    # ---------------------------------------------------
    # PROPERTY INSIGHTS
    # ---------------------------------------------------
    st.subheader("📊 Property Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "City",
            city
        )

    with col2:
        st.metric(
            "Location",
            location
        )

    with col3:
        st.metric(
            "Luxury Category",
            luxury_category
        )

    # ---------------------------------------------------
    # DETAILS
    # ---------------------------------------------------
    st.write("### 🏡 Property Summary")

    st.write(f"📏 Area: **{area} sqft**")

    st.write(f"🛏️ Bedrooms: **{bedrooms}**")

    st.write(f"⭐ Luxury Score: **{luxury_score:.2f}**")

    # ---------------------------------------------------
    # MARKET INSIGHTS
    # ---------------------------------------------------
    st.write("### 📈 Market Insights")

    if city == "Mumbai":

        st.info(
            "Mumbai properties generally have higher price per sqft and compact luxury housing."
        )

    else:

        st.info(
            "Delhi properties generally provide larger living spaces at lower price per sqft."
        )

    # ---------------------------------------------------
    # PRICE CATEGORY
    # ---------------------------------------------------
    if predicted_price < 5000000:

        st.warning("💰 Budget-Friendly Property")

    elif predicted_price < 15000000:

        st.info("🏢 Mid-Range Property")

    else:

        st.success("🌟 Premium Luxury Property")