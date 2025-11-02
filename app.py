"""
Problem Statement:
This app predicts product sales based on advertising budgets for TV, Radio, and Newspaper using a multiple linear regression model trained on the Advertising dataset.

Instructions:
Enter the advertising budgets for TV, Radio, and Newspaper below and click 'Predict' to see the estimated sales.
"""

# Import necessary libraries
import streamlit as st
import numpy as np
import joblib
import os

# Set page title and description
st.title("Advertising Sales Prediction App")
st.write("Predict product sales based on advertising budgets for TV, Radio, and Newspaper.")

# Sidebar for user input
st.sidebar.header("Input Advertising Budgets")
tv = st.sidebar.number_input("TV Budget ($)", min_value=0.0, value=100.0, step=10.0)
radio = st.sidebar.number_input("Radio Budget ($)", min_value=0.0, value=20.0, step=1.0)
newspaper = st.sidebar.number_input("Newspaper Budget ($)", min_value=0.0, value=10.0, step=1.0)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()

# Predict button
if st.button("Predict"):
    # Prepare input for prediction
    input_data = np.array([[tv, radio, newspaper]])
    # Make prediction
    prediction = model.predict(input_data)
    st.success(f"Predicted Sales: {prediction[0]:.2f} units")

# Add footer
st.markdown("---")
st.markdown("Multiple Linear Regression Example")
