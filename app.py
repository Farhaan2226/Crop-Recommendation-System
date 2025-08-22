import streamlit as st
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import pandas as pd
import streamlit.components.v1 as components

# Load models and tools
rf = pickle.load(open("random_forest_model.pkl", "rb"))
knn = pickle.load(open("knn_model.pkl", "rb"))
xgb = pickle.load(open("xgboost_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.set_page_config(page_title="Crop Recommendation", layout="centered")

st.title("🌾 Crop Recommendation System")
st.write("Enter your soil and weather details to get the best crop suggestion.")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

N = st.sidebar.slider("Nitrogen (N)", 0, 140, 80)
P = st.sidebar.slider("Phosphorus (P)", 5, 145, 40)
K = st.sidebar.slider("Potassium (K)", 5, 205, 40)
temperature = st.sidebar.slider("Temperature (°C)", 10.0, 45.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 70.0)
ph = st.sidebar.slider("pH", 3.5, 9.5, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "KNN", "XGBoost"])

if st.sidebar.button("Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)

    model = {"Random Forest": rf, "KNN": knn, "XGBoost": xgb}[model_choice]
    prediction = model.predict(input_scaled)[0]
    crop_name = le.inverse_transform([prediction])[0]

    st.success(f"🌱 Recommended Crop: **{crop_name.capitalize()}**")

    # SHAP explanation for tree models
    if model_choice in ["Random Forest", "XGBoost"]:
        explainer = shap.Explainer(model)
        shap_values = explainer(input_scaled)

        pred_class = prediction
        if hasattr(model, 'classes_'):
            pred_class_idx = list(model.classes_).index(prediction)
        else:
            pred_class_idx = prediction

      

    # Simple feature input bar chart (alternative explanation)
    features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    values = [N, P, K, temperature, humidity, ph, rainfall]
    input_df = pd.DataFrame({'Feature': features, 'Value': values})

    st.subheader("📊 Input Feature Values")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(input_df['Feature'], input_df['Value'], color='mediumseagreen')
    ax2.set_xlabel('Value')
    ax2.set_title('Input Feature Values')
    st.pyplot(fig2)
