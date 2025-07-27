import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import preprocess_signal, predict_ecg, generate_saliency_map

# Load trained model
model = load_model("lstm_model.keras")
# ECG plot helper
def plot_ecg(signal, saliency=None):
    fig, ax = plt.subplots()
    ax.plot(signal, label="ECG Signal", color="blue")
    if saliency is not None:
        ax.plot(saliency, label="Saliency", color="red", linestyle="--")
    ax.set_title("ECG Signal with Saliency")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend()
    return fig

# Main logic
def run_prediction(input_file, manual_input, sample_choice):
    signal = None

    # Load from sample
    if sample_choice == "Normal Sample":
        df = pd.read_csv("Sample_data/1_normal_beat.csv", header=None)
        signal = df.values.flatten()
        signal = pd.to_numeric(signal, errors="coerce")
        signal = signal[~np.isnan(signal)]
        if len(signal) == 360:
            start = (360 - 180) // 2
            signal = signal[start: start + 180]

    elif sample_choice == "Abnormal Sample":
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

    # Load from uploaded file
    elif input_file is not None:
        df = pd.read_csv(input_file, header=None)
        if df.shape[0] >= 2:
            signal = df.iloc[1].values
            signal = pd.to_numeric(signal, errors='coerce')
            signal = signal[~np.isnan(signal)]
            if len(signal) == 180:
                signal = signal.astype(np.float32)
            else:
                return "❌ Uploaded file must contain 180 values in second row.", None, None

    # Load from manual input
    elif manual_input.strip():
        try:
            signal = [float(x.strip()) for x in manual_input.split(',') if x.strip()]
            if len(signal) != 180:
                return "❌ Manual input must have exactly 180 values.", None, None
        except Exception as e:
            return f"❌ Invalid manual input: {e}", None, None

    if signal is None or len(signal) != 180:
        return "❌ Invalid ECG Signal. Make sure input has 180 values.", None, None

    # Prediction & plotting
    signal = np.array(signal, dtype=np.float32)
    processed = preprocess_signal(signal)
    pred_label, prob = predict_ecg(model, processed)
    saliency = generate_saliency_map(model, processed)

    fig = plot_ecg(signal, saliency)

    return f"🧠 Prediction: {pred_label} ", fig, fig

# Gradio UI
iface = gr.Interface(
    fn=run_prediction,
    inputs=[
        gr.File(label="Upload ECG CSV File", type="filepath"),
        gr.Textbox(label="Or paste 180 comma-separated ECG values"),
        gr.Radio(["None", "Normal Sample", "Abnormal Sample"], label="Or Load Sample", value="None")
    ],
    outputs=[
        gr.Text(label="Prediction"),
        gr.Plot(label="ECG + Saliency Plot"),
        gr.Plot(visible=False)  # legacy; can be used for separate saliency if needed
    ],
    title="🔍 ECG Anomaly Detection with Saliency Map",
    description="Upload a CSV or paste 180 ECG values to detect abnormalities using a trained LSTM model. A saliency map explains which parts of the ECG were important in the prediction."
)

if __name__ == "__main__":
    iface.launch()
