# 🧠 ANN-Application - Customer Churn Prediction

This project is a part of Generative AI Complete Bootcamp - NLP, Transformers & Gen AI in Udemy.
This project is a **Streamlit-based web application** that predicts whether a bank customer is likely to churn or stay, using a trained **Artificial Neural Network (ANN)** built with **TensorFlow Keras**.

---

## 🔧 Features
- Real-time prediction of customer churn via web UI.
- Encodes categorical variables (`Gender`, `Geography`) using pre-trained encoders.
- Scales numerical inputs with pre-fitted `StandardScaler`.
- Accepts user inputs interactively through sliders, dropdowns, and number inputs.
- Displays:
  - Churn probability (e.g. `0.83`)
  - Prediction message (e.g. "Customer is likely to churn")
  - Visual progress bar for intuitive understanding.

---

## 🚀 How to Run Locally

### 1. 📦 Clone the repository

git clone https://github.com/Muzann11/ANN-Application.git
cd ANN-Application

### 2. 🐍 Set up a virtual environment (recommended)

> Note: Use Python **3.11.x** (tested with TensorFlow 2.15)

python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux


### 3. 📄 Install dependencies

pip install -r requirements.txt

Or manually:

pip install streamlit tensorflow==2.15 scikit-learn pandas

### 4. 🧪 Run the app

streamlit run ANN_WebApp.py

---

## 📁 Project Structure

```
ANN-Application/
├── ANN_WebApp.py              # Streamlit web app
├── trained_model.keras        # Saved ANN model (excluded from Git)
├── dataScaler.pkl             # StandardScaler object
├── gender_encoder.pkl         # LabelEncoder for Gender
├── geography_encoder.pkl      # OneHotEncoder for Geography
├── Data_Modeling.ipynb        # Jupyter notebook for model training
├── Predict_Customer_Churn.ipynb  # Jupyter notebook for testing
└── README.md                  # This file
```

---

## 📌 Requirements

* Python 3.11
* TensorFlow 2.15
* Streamlit 1.x
* scikit-learn
* pandas

---

## 💡 Notes

* `.pkl` files must match the encoders and scaler used during training.
* `.keras` or `.h5` model must be the same architecture expected in the app.
* Model files are typically large and **excluded** from the repository (you can store them in cloud or Hugging Face if needed).

---

