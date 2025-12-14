import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Employee retention prediction",
    page_icon="💳",
    layout="centered"
)

# Load model & encoders
model = joblib.load("xgboost_fraud_model.pkl")
encoders = joblib.load("label_encoders.pkl")

st.markdown(
    "<h1 style='text-align:center;'>Employee Retention Prediction</h1>",
    unsafe_allow_html=True
)

st.divider()

st.subheader("Enter Candidate Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", encoders["gender"].classes_)
    relevent_experience = st.selectbox(
        "Relevant Experience", encoders["relevent_experience"].classes_
    )
    enrolled_university = st.selectbox(
        "University Enrollment", encoders["enrolled_university"].classes_
    )
    education_level = st.selectbox(
        "Education Level", encoders["education_level"].classes_
    )
    major_discipline = st.selectbox(
        "Major Discipline", encoders["major_discipline"].classes_
    )

with col2:
    experience = st.selectbox(
        "Years of Experience", encoders["experience"].classes_
    )
    company_size = st.selectbox(
        "Company Size", encoders["company_size"].classes_
    )
    company_type = st.selectbox(
        "Company Type", encoders["company_type"].classes_
    )
    last_new_job = st.selectbox(
        "Years Since Last Job Change", encoders["last_new_job"].classes_
    )
    training_hours = st.number_input(
        "Training Hours", min_value=0, max_value=500, value=20
    )

# Build input
input_df = pd.DataFrame([{
    "gender": gender,
    "relevent_experience": relevent_experience,
    "enrolled_university": enrolled_university,
    "education_level": education_level,
    "major_discipline": major_discipline,
    "experience": experience,
    "company_size": company_size,
    "company_type": company_type,
    "last_new_job": last_new_job,
    "training_hours": training_hours
}])

# Encode
for col, le in encoders.items():
    input_df[col] = le.transform(input_df[col])

st.divider()

if st.button("🔍 Predict Fraud", use_container_width=True):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ Fraud Detected (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ Not Fraud (Confidence: {(1 - prob):.2%})")

st.divider()
st.caption("Built By FARHAN with XGBoost • Streamlit • ML")
