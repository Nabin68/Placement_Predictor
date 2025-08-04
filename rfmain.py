import streamlit as st
import pandas as pd
import joblib

model1 = joblib.load("classification_model.pkl")
model2 = joblib.load("regression_model.pkl")

st.title("Placement Prediction App 🎓💼")

gender = st.selectbox("Gender", ["M", "F"])
ssc_b = st.selectbox("SSC Board", ["Central", "Others"])
hsc_b = st.selectbox("HSC Board", ["Central", "Others"])
hsc_s = st.selectbox("HSC Stream", ["Science", "Commerce", "Arts"])
degree_t = st.selectbox("Degree Stream", ["Sci&Tech", "Comm&Mgmt", "Others"])
degree_p = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0)
workex = st.selectbox("Work Experience", ["Yes", "No"])
specialisation = st.selectbox("MBA Specialisation", ["Mkt&Fin", "Mkt&HR"])

ssc_p = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0)
hsc_p = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0)
etest_p = st.number_input("E-Test Percentage", min_value=0.0, max_value=100.0)

if st.button("Predict Placement Status"):
    input_df = pd.DataFrame({
        "gender": [gender],
        "ssc_b": [ssc_b],
        "hsc_b": [hsc_b],
        "hsc_s": [hsc_s],
        "degree_t": [degree_t],
        "degree_p": [degree_p],
        "workex": [workex],
        "specialisation": [specialisation],
        "ssc_p": [ssc_p],
        "hsc_p": [hsc_p],
        "etest_p": [etest_p]
    })

    prediction = model1.predict(input_df)[0]
    prediction_label = "Placed ✅" if prediction == 1 else "Not Placed ❌"
    st.subheader(f"Prediction: {prediction_label}")

    if prediction == 1:
        input_reg_df = input_df.copy()
        input_reg_df["status"] = ["Placed"]
        salary = model2.predict(input_reg_df)[0]
        st.subheader(f"Estimated Salary: ₹{int(salary):,}")
    else:
        st.info("Salary prediction is only available for placed candidates.")
