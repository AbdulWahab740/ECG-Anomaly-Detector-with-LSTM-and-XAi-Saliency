import numpy as np

def preprocess_signal(signal):
    mean = np.mean(signal)
    std = np.std(signal) + 1e-6
    normalized_signal = (signal - mean) / std

        # Reshape to match model input: (1, signal_length)
    input_signal = normalized_signal.reshape(1, -1)

    signal = np.array(input_signal).reshape(1, -1)  # e.g., (1, 180)
    # Add normalization if needed
    return signal

def predict_ecg(model, signal):
    prediction = model.predict(signal)

    # Flatten prediction if needed
    if prediction.ndim > 1:
        prob = prediction[0][0]  # Assuming shape (1, 1)
    else:
        prob = prediction[0]

    # Since 1 = Abnormal, use threshold accordingly
    predicted_class = "Abnormal" if prob >= 0.5 else "Normal"
    confidence = float(prob if predicted_class == "Abnormal" else 1 - prob) * 100

    return predicted_class, confidence

def generate_saliency_map(model, signal):
    # Dummy version (you can replace with LIME/SHAP/saliency methods)
    # For real DL models use torch/tf gradient-based xAI
    weights = np.random.uniform(0.1, 1.0, size=signal.shape[1])
    return weights
