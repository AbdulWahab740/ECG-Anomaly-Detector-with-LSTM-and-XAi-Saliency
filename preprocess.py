import numpy as np

def normalize_segment(segment):
    mean = np.mean(segment)
    std = np.std(segment) + 1e-6
    return (segment - mean) / std

def preprocess_uploaded_ecg(ecg_signal):
    """
    Preprocesses a 1D ECG signal uploaded by the user.

    Parameters:
    - ecg_signal (list or np.ndarray): raw ECG signal values

    Returns:
    - X (np.ndarray): preprocessed ECG segment, shape (1, 180, 1)
    """
    ecg_signal = np.array(ecg_signal)

    if len(ecg_signal) < 180:
        raise ValueError("ECG signal must be at least 180 samples long.")

    start = (len(ecg_signal) // 2) - 90
    end = (len(ecg_signal) // 2) + 90
    segment = ecg_signal[start:end]

    segment = normalize_segment(segment)

    return segment.reshape(1, 180, 1)
