# ==========================================
# House Price Prediction App
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from utils import create_input_dataframe, preprocess_input

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="🏡 House Price Prediction",
    page_icon="assets/banner.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Load Trained Files
# ==========================================

try:
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

except Exception as e:
    st.error(f"Error loading model files:\n{e}")
    st.stop()


# ==========================================
# Load Original Dataset
# ==========================================

housing_df = pd.read_csv("data/AmesHousing.csv")


# ==========================================
# Custom CSS
# ==========================================

st.markdown("""
<style>

.main{
    background-color:#F7F9FC;
}

h1{
    color:#1E3A8A;
    text-align:center;
}

.stButton>button{
    width:100%;
    background:#2563EB;
    color:white;
    border-radius:10px;
    height:3em;
    font-size:18px;
}

.stButton>button:hover{
    background:#1D4ED8;
}

.metric-card{
    padding:20px;
    border-radius:10px;
    background:white;
    box-shadow:0px 2px 8px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# Title
# ==========================================

st.markdown(
    """
    <h1 style='text-align:center; color:#2E86C1;'>
        🏡 House Price Prediction System
    </h1>
    <h4 style='text-align:center; color:gray;'>
        Predict residential house prices using Machine Learning
    </h4>
    <hr>
    """,
    unsafe_allow_html=True
)

# ==========================================
# Sidebar
# ==========================================

with st.sidebar:

    st.image("assets/house.png", width=110)

    st.title("🏡 Price Predictor")

    st.markdown("---")

    st.subheader("📌 Project Information")

    st.info("""
**🤖 Model**
- Random Forest Regressor

**📊 Dataset**
- Ames Housing Dataset

**🏠 Original Features**
- 82

**🎯 Prediction Features**
- 25

**📈 Model R² Score**
- 0.904
""")

    st.markdown("---")

    st.subheader("📝 How to Use")

    st.markdown("""
1️⃣ Enter the property details.

2️⃣ Click **Predict Price**.

3️⃣ View the estimated selling price.
""")

    st.markdown("---")

    st.success("✅ Built with Streamlit + Scikit-Learn")

st.markdown("## 🏠 Property Information")

st.write(
    "Provide the following house details to estimate the selling price."
)

st.markdown("---")

# ==========================================
# General Information
# ==========================================

with st.expander("🏠 General Information", expanded=True):

    col1, col2 = st.columns(2)

    with col1:

        overall_qual = st.slider(
            "Overall Quality",
            min_value=1,
            max_value=10,
            value=5
        )

        year_built = st.number_input(
            "Year Built",
            min_value=1872,
            max_value=2025,
            value=2000
        )

    with col2:

        overall_cond = st.slider(
            "Overall Condition",
            min_value=1,
            max_value=10,
            value=5
        )

        year_remod = st.number_input(
            "Year Remod/Add",
            min_value=1950,
            max_value=2025,
            value=2000
        )


# ==========================================
# Property Size
# ==========================================

with st.expander("📐 Property Size", expanded=True):

    col1, col2 = st.columns(2)

    with col1:

        lot_area = st.number_input(
            "Lot Area (sq ft)",
            min_value=1000,
            max_value=250000,
            value=10000,
            step=100
        )

        gr_liv_area = st.number_input(
            "Above Ground Living Area (sq ft)",
            min_value=300,
            max_value=6000,
            value=1500,
            step=50
        )

        garage_area = st.number_input(
            "Garage Area (sq ft)",
            min_value=0,
            max_value=2000,
            value=500,
            step=10
        )

    with col2:

        lot_frontage = st.number_input(
            "Lot Frontage (ft)",
            min_value=0,
            max_value=350,
            value=70,
            step=1
        )

        total_bsmt_sf = st.number_input(
            "Total Basement Area (sq ft)",
            min_value=0,
            max_value=7000,
            value=1000,
            step=50
        )

        garage_cars = st.selectbox(
            "Garage Capacity (Cars)",
            options=[0, 1, 2, 3, 4, 5],
            index=2
        )


# ==========================================
# Rooms & Amenities
# ==========================================

with st.expander("🛏️ Rooms & Amenities", expanded=True):

    col1, col2 = st.columns(2)

    with col1:

        full_bath = st.selectbox(
            "Full Bathrooms",
            options=[0, 1, 2, 3, 4],
            index=2
        )

        bedroom_abvgr = st.selectbox(
            "Bedrooms Above Ground",
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            index=2
        )

        fireplaces = st.selectbox(
            "Number of Fireplaces",
            options=[0, 1, 2, 3],
            index=1
        )

    with col2:

        half_bath = st.selectbox(
            "Half Bathrooms",
            options=[0, 1, 2],
            index=0
        )

        totrms_abvgrd = st.selectbox(
            "Total Rooms Above Ground",
            options=list(range(2, 16)),
            index=4
        )


# ==========================================
# Quality Features
# ==========================================

with st.expander("⭐ Quality Features", expanded=True):

    col1, col2 = st.columns(2)

    quality_options = ["Po", "Fa", "TA", "Gd", "Ex"]

    with col1:

        kitchen_qual = st.selectbox(
            "Kitchen Quality",
            options=quality_options,
            index=2
        )

        heating_qc = st.selectbox(
            "Heating Quality",
            options=quality_options,
            index=2
        )

    with col2:

        exter_qual = st.selectbox(
            "Exterior Quality",
            options=quality_options,
            index=2
        )


# ==========================================
# Property Type
# ==========================================

with st.expander("🏘️ Property Type", expanded=True):

    col1, col2 = st.columns(2)

    neighborhood_options = sorted(housing_df["Neighborhood"].dropna().unique())
    house_style_options = sorted(housing_df["House Style"].dropna().unique())
    bldg_type_options = sorted(housing_df["Bldg Type"].dropna().unique())
    foundation_options = sorted(housing_df["Foundation"].dropna().unique())

    with col1:

        neighborhood = st.selectbox(
            "Neighborhood",
            options=neighborhood_options
        )

        house_style = st.selectbox(
            "House Style",
            options=house_style_options
        )

    with col2:

        bldg_type = st.selectbox(
            "Building Type",
            options=bldg_type_options
        )

        foundation = st.selectbox(
            "Foundation",
            options=foundation_options
        )               


# ==========================================
# Basement & Utilities
# ==========================================

with st.expander("🏗️ Basement & Utilities", expanded=True):

    col1, col2 = st.columns(2)

    basement_quality = [
        "None",
        "Po",
        "Fa",
        "TA",
        "Gd",
        "Ex"
    ]

    garage_finish_options = [
        "None",
        "Unf",
        "RFn",
        "Fin"
    ]

    central_air_options = [
        "N",
        "Y"
    ]

    with col1:

        bsmt_qual = st.selectbox(
            "Basement Quality",
            options=basement_quality,
            index=3
        )

        central_air = st.selectbox(
            "Central Air",
            options=central_air_options,
            index=1
        )

    with col2:

        garage_finish = st.selectbox(
            "Garage Finish",
            options=garage_finish_options,
            index=2
        )                         


# ==========================================
# Collect User Inputs
# ==========================================

user_inputs = {

    # General Information
    "Overall Qual": overall_qual,
    "Overall Cond": overall_cond,
    "Year Built": year_built,
    "Year Remod/Add": year_remod,

    # Property Size
    "Lot Area": lot_area,
    "Lot Frontage": lot_frontage,
    "Gr Liv Area": gr_liv_area,
    "Total Bsmt SF": total_bsmt_sf,
    "Garage Area": garage_area,
    "Garage Cars": garage_cars,

    # Rooms
    "Full Bath": full_bath,
    "Half Bath": half_bath,
    "Bedroom AbvGr": bedroom_abvgr,
    "TotRms AbvGrd": totrms_abvgrd,
    "Fireplaces": fireplaces,

    # Quality
    "Kitchen Qual": kitchen_qual,
    "Exter Qual": exter_qual,
    "Heating QC": heating_qc,

    # Property Type
    "Neighborhood": neighborhood,
    "House Style": house_style,
    "Bldg Type": bldg_type,
    "Foundation": foundation,

    # Basement & Utilities
    "Bsmt Qual": bsmt_qual,
    "Garage Finish": garage_finish,
    "Central Air": central_air
}


st.markdown("---")

predict_button = st.button(
    "🏡 Predict House Price",
    use_container_width=True
)


if predict_button:

    # Create DataFrame
    input_df = create_input_dataframe(user_inputs)

    
    

    # Preprocess
    processed_df = preprocess_input(
        input_df,
        feature_columns
    )

    
    

    # Scale
    scaled_data = scaler.transform(processed_df)

    
    

    # Prediction
    prediction = model.predict(scaled_data)[0]

    st.success("Prediction completed successfully!")

    st.metric(
        label="Estimated House Price",
        value=f"${prediction:,.0f}"
    )


# ==========================================
# Footer
# ==========================================

st.markdown("---")

st.caption(
    "Developed by Kasi Rajan | Powered by Streamlit & Scikit-learn"
)