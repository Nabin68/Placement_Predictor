# 🎓 Placement Prediction App 💼

A web-based machine learning application that predicts whether a student will be placed and estimates the salary based on academic and personal attributes using classification and regression models.


## 🚀 Project Overview

This project uses a combination of **Random Forest Classifier** and **Random Forest Regressor** with **GridSearchCV** to:

- Predict **placement status** (Placed or Not Placed) — _classification_
- Estimate **expected salary** for placed students — _regression_

The app is deployed using **Streamlit** and allows users to input academic records and get instant predictions.

---

## 📊 Model Performance

### Classification
- **Accuracy**: `0.8139`
- **Best Parameters**:
  ```python
  {
      'classification_model__n_estimators': 50,
      'classification_model__max_samples': 1.0,
      'classification_model__max_features': 1.0
  }
  ```

### Regression
- **Root Mean Squared Error (RMSE)**: `90118.33`
- **Best Parameters**:
  ```python
  {
      'regression_model__n_estimators': 50,
      'regression_model__max_samples': 1.0,
      'regression_model__max_features': 0.5
  }
  ```

---

## 🧠 Technologies Used

- Python
- scikit-learn
- pandas, numpy
- Streamlit (for web interface)
- joblib (for model persistence)

---

## 📁 Files

- `randomforest.py` – Core training script for classification and regression  
- `classification_model.pkl` & `regression_model.pkl` – Saved models used in the app  
- `app.py` – Streamlit-based web app for predictions  
- `Placement_Data_Full_Class.csv` – Original dataset used for model training  
- `screenshot.png` – UI screenshot of the Streamlit app  
- `README.md` – Project documentation

---

## ⚙️ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nabin68/Placement-Predictor.git
   cd placement-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 💾 Notes

- `.pkl` model files are included for future use and reproducibility.
- You can retrain models using `randomforest.py` if needed.
- Ensure `Placement_Data_Full_Class.csv` and `screenshot.png` are present in the root directory.
