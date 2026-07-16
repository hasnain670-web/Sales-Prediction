import streamlit as st
import joblib
import pandas as pd

# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Sales Prediction",
    page_icon="📈",
    layout="centered"
)

# --------------------------------
# Load Model
# --------------------------------

model = joblib.load("models/sales_model.pkl")

# --------------------------------
# Sidebar
# --------------------------------

st.sidebar.title("📊 Sales Prediction")

st.sidebar.info(
    """
    Predict product sales based on
    advertising budgets.

    Model:
    Linear Regression

    Author:
    Muhammad Hasnain
    """
)

# --------------------------------
# Title
# --------------------------------

st.title("📈 Sales Prediction App")

st.write(
    "Enter the advertising budgets below to predict product sales."
)

st.divider()

# --------------------------------
# User Inputs
# --------------------------------

tv = st.number_input(
    "📺 TV Advertisement Budget",
    min_value=0.0,
    value=150.0,
    step=1.0
)

radio = st.number_input(
    "📻 Radio Advertisement Budget",
    min_value=0.0,
    value=25.0,
    step=1.0
)

newspaper = st.number_input(
    "📰 Newspaper Advertisement Budget",
    min_value=0.0,
    value=20.0,
    step=1.0
)

# --------------------------------
# Predict Button
# --------------------------------

if st.button("Predict Sales", use_container_width=True):

    input_data = [[tv, radio, newspaper]]

    prediction = model.predict(input_data)

    st.success(
        f"📈 Predicted Sales: {prediction[0]:.2f} Thousand Units"
    )

    st.divider()

    st.subheader("📋 Input Summary")

    st.write(f"📺 TV Budget : {tv}")
    st.write(f"📻 Radio Budget : {radio}")
    st.write(f"📰 Newspaper Budget : {newspaper}")

    st.divider()

    st.subheader("📊 Advertisement Budget Comparison")

    chart_data = pd.DataFrame({
        "Advertising": ["TV", "Radio", "Newspaper"],
        "Budget": [tv, radio, newspaper]
    })

    st.bar_chart(
        chart_data.set_index("Advertising")
    )

    st.divider()

    st.subheader("💡 Business Insight")

    if prediction[0] >= 20:
        st.success(
            "Excellent! Expected sales are very high."
        )

    elif prediction[0] >= 10:
        st.info(
            "Good! Expected sales are moderate."
        )

    else:
        st.warning(
            "Low expected sales. Consider increasing your advertising budget."
        )