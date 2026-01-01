# =====================================================
# PATH SETUP
# =====================================================
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

MODEL_DIR = os.path.join(BASE_DIR, "Model")

# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.write("Fraud detection using a Balanced Random Forest model (High Recall Mode).")

# =====================================================
# LOAD MODEL + FEATURES
# =====================================================
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "brf_final_model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "brf_features.pkl"))
    return model, features

brf_model, training_features = load_model()

# =====================================================
# FEATURE ALIGNMENT
# =====================================================
def align_features(df, feature_list):
    df = df.copy()
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0
    return df[feature_list]

# =====================================================
# INPUT METHOD
# =====================================================
st.subheader("📥 Input Method")

input_method = st.radio(
    "Choose how you want to provide data:",
    ("Upload CSV File", "Manual Input")
)

# =====================================================
# OPTION 1: CSV INPUT
# =====================================================
if input_method == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("🔍 Predict Fraud (CSV)"):
            df_aligned = align_features(df, training_features)

            probs = brf_model.predict_proba(df_aligned)[:, 1]
            df["Fraud_Risk_Score"] = probs
            df["Prediction"] = np.where(
                probs >= 0.35,
                "🚨 Fraudulent",
                "✅ Legitimate"
            )

            st.success("Prediction completed!")
            st.dataframe(df.head(10))

            st.metric(
                "🚨 Flagged Transactions",
                int((probs >= 0.35).sum())
            )

# =====================================================
# OPTION 2: MANUAL INPUT
# =====================================================
if input_method == "Manual Input":
    st.subheader("✍️ Enter Transaction Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        amt = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
        city_pop = st.number_input("City Population", min_value=0, value=50000)
        unix_time = st.number_input("Unix Time", min_value=0, value=1546300800)

    with col2:
        lat = st.number_input("Latitude", value=30.0)
        long = st.number_input("Longitude", value=31.0)
        merch_lat = st.number_input("Merchant Latitude", value=30.05)

    with col3:
        merch_long = st.number_input("Merchant Longitude", value=31.05)
        gender = st.selectbox("Gender", ["Male", "Female"])
        gender_encoded = 1 if gender == "Male" else 0

    # =================================================
    # RAW INPUT DATAFRAME
    # =================================================
    input_df = pd.DataFrame([{
        "amt": amt,
        "city_pop": city_pop,
        "unix_time": unix_time,
        "lat": lat,
        "long": long,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "gender": gender_encoded
    }])

    # ===== DERIVED FEATURES =====
    input_df["amt_log"] = np.log1p(input_df["amt"])
    input_df["age"] = 30
    input_df["amt_zscore"] = 0

    st.write("Input Preview:")
    st.dataframe(input_df)

    if st.button("🔍 Predict Fraud (Manual)"):
        input_df_aligned = align_features(input_df, training_features)
        fraud_prob = brf_model.predict_proba(input_df_aligned)[0][1]

        st.metric("Fraud Risk Score", f"{fraud_prob:.2f}")

        if fraud_prob >= 0.35:
            st.error("🚨 Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")
