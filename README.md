#  ECG Classification & Anomaly Detection with LSTM & Explainable AI (XAI)

This project focuses on detecting abnormal ECG patterns using an LSTM-based deep learning model, trained on real-world physiological signals from the PhysioNet dataset. It further applies **Explainable AI** (XAI) techniques to visualize and interpret model predictions, providing transparency and trust—especially critical for medical applications.

---

## 📌 Objectives

- Classify ECG signals as **Normal** or **Abnormal**
- Use **LSTM** networks to model temporal patterns
- Apply **Explainable AI (XAI)** to interpret decisions
- Visualize and highlight the **important time steps**

---

## 📂 Dataset

- **Source**: [PhysioNet MLTDB ECG Dataset](https://physionet.org/)
- **Signals**: Two-lead ECG data (`MLII`, `V5`)
- **Labels**: Annotated with heartbeat types (`N`, `A`, etc.)
- **Preprocessing**:
  - Extracted 180-sample ECG windows
  - Labeled windows based on central beat
  - Balanced classes (Normal vs Abnormal)

---

## 🧠 Model Architecture

```
Input Shape: (180, 1)

Layer Stack:
1. LSTM (256 units) → return_sequences=True
2. Dropout (0.4)
3. LSTM (256 units)
4. Dense (32, relu)
5. Dropout (0.2)
6. Dense (1, sigmoid)
Optimizer: Adam (lr = 0.0001)

Loss: Binary Crossentropy

EarlyStopping: Enabled (patience=5)
```

## 📈 Performance
```
Metric	Value
Accuracy	98%
F1-Score	0.97
AUC	0.98
Samples	~110k
```

## 📉 Confusion Matrix

![Alt Text](/Outputs/confusion-matrix.png)

## 🔍 Explainable AI (XAI)
I used Saliency Maps to identify which ECG time points influenced the model’s decision the most.

## 🖼️ Example Visualization:

![Alt Text](/Outputs/important-points-in-ecg-with-saliency-overlay.png)

Red highlights show which parts of the ECG the model found important

Helps verify if the model relies on relevant medical patterns

## 🛠️ Tech Stack
```
Python (3.11+)
TensorFlow / Keras
WFDB + PhysioNet
tf-keras-vis
Matplotlib / Seaborn
Streamlit
```

## 🙋‍♂️ Author

Created by [Abdul Wahab](https://linkedin.com/in/abwahab07)

Machine Learning & AI Enthusiast
[Github](https://github.com/AbdulWahab740)


---

Would you like a `requirements.txt` and `train.py` template to go with it? Or shall we move forward with building a **Streamlit UI** for this model?
