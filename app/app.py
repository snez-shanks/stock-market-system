import sys
import os

# Fix import issue for src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd

from src.load_data import load_data
from src.model import train_model


# =========================
# PAGE TITLE
# =========================
st.title("📈 Stock Market Analysis & Prediction System")


# =========================
# LOAD DATA
# =========================
st.subheader("📊 Stock Data")

try:
    df = load_data()

    if df is not None:
        st.dataframe(df.head(50))  # show first 50 rows
    else:
        st.error("No data loaded.")

except Exception as e:
    st.error(f"Error loading data: {e}")


# =========================
# BASIC STATS
# =========================
if df is not None:
    st.subheader("📈 Data Summary")
    st.write(df.describe())


# =========================
# GRAPH
# =========================
if df is not None:
    st.subheader("📉 Stock Price Trend")
    st.line_chart(df['Adj Close'])


# =========================
# MODEL TRAINING
# =========================
st.subheader("🤖 Train Models")

if st.button("Run Prediction Models"):
    st.info("Training models... please wait ⏳")

    try:
        models = train_model()

        st.success("✅ Models trained successfully!")

        st.write("### Models Included:")
        st.write("✔ Linear Regression")
        st.write("✔ Logistic Regression")
        st.write("✔ Random Forest")
        st.write("🔥 XGBoost (Best Model)")

    except Exception as e:
        st.error(f"Error during model training: {e}")