import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import preprocess_signal, predict_ecg, generate_saliency_map

# Load model
model = load_model('model/LSTM on ECG Anomaly Detector.keras')

st.title("🔍 ECG Anomaly Detection with Saliency Map")

st.write("Upload a CSV file (1 row with 180 ECG values) or paste comma-separated values below:")

# Upload CSV
uploaded_file = st.file_uploader("Upload ECG CSV", type="csv")

# Session state to store ECG data if loaded from button
if "manual_signal" not in st.session_state:
    st.session_state.manual_signal = None

# Text input
manual_input_str = st.text_area("Or paste 180 comma-separated ECG values")

with st.sidebar:
    if st.button("Load Normal Sample"):
        df = pd.read_csv("Sample_data/1_normal_beat.csv", header=None)
        signal = df.values.flatten()
        signal = pd.to_numeric(signal, errors='coerce')
        signal = signal[~np.isnan(signal)]
        if len(signal) == 360:
            start = (360 - 180) // 2
            signal = signal[start: start + 180]
        if len(signal) != 180:
            st.error(f"Normal sample has {len(signal)} values instead of 180.")
        else:
            st.session_state.manual_signal = signal.tolist()
            st.success("Loaded the Normal Sample! Click Predict.")

    if st.button("Load Abnormal Sample"):
        df = pd.read_csv("Sample_data/1_abnormal_beat.csv", header=None)
        flat = df.values.flatten()
        signal = []
        for x in flat:
            try:
                val = float(str(x).strip())
                signal.append(val)
            except:
                continue
        if len(signal) == 360:
            start = (360 - 180) // 2
            signal = signal[start: start + 180]
        elif len(signal) == 181:
            signal = signal[:-1]
        if len(signal) != 180:
            st.error(f"Abnormal sample has {len(signal)} values instead of 180.")
        else:
            st.session_state.manual_signal = signal
            st.success("Loaded the Abnormal Sample! Click Predict.")

# Parse manual input if available
manual_signal = st.session_state.manual_signal
if not manual_signal and manual_input_str:
    try:
        manual_signal = [float(x.strip()) for x in manual_input_str.split(',') if x.strip() != '']
        if len(manual_signal) != 180:
            st.error(f"Manual input has {len(manual_signal)} values instead of 180.")
            manual_signal = None
    except Exception as e:
        st.error(f"Invalid input format! Error: {e}")
        manual_signal = None

# Run prediction
if st.button("Predict!"):
    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=None)
        if df.shape[0] < 2:
            st.error("CSV must have at least 2 rows (header and signal values).")
            signal = None
        else:
            signal = df.iloc[1].values
            signal = pd.to_numeric(signal, errors='coerce')
            signal = signal[~np.isnan(signal)]
            if len(signal) != 180:
                st.error(f"Expected 180 values, but got {len(signal)}.")
                signal = None
            else:
                signal = signal.astype(np.float32)
    elif manual_signal:
        signal = np.array(manual_signal, dtype=np.float32)
    else:
        signal = None

    if signal is not None and len(signal) == 180:
        st.success("✅ ECG Signal Loaded")
        processed = preprocess_signal(signal)

        pred_label, prob = predict_ecg(model, processed)
        
        st.markdown(f"### 🧠 Prediction: **{pred_label}**")

        st.subheader("📈 ECG Signal")
        st.line_chart(signal)

        saliency = generate_saliency_map(model, processed)

        st.subheader("🔍 Saliency Map")
        fig, ax = plt.subplots()
        ax.plot(signal, label="ECG Signal", color="blue")
        ax.plot(saliency, label="Saliency", color="red", linestyle="--")
        ax.legend()
        st.pyplot(fig)

        st.markdown("""
        ### 🧠 What is a Saliency Map?

        A saliency map shows **which parts of the ECG signal were most important** in the model's prediction.  
        - It helps explain **why** the model predicted *Normal* or *Abnormal*.
        - Peaks in the red dashed line (saliency) mean those parts of the ECG were **most influential** in the decision.

        This makes the model's behavior **more transparent and trustworthy**, especially in healthcare.
        """)
    else:
        st.info("Please upload or paste a valid 180-value ECG signal.")
