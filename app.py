import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/best_model.pkl")

st.title("🎓 Student Performance Prediction System")

st.write("""
This application predicts whether a student will Pass or Fail
based on academic performance indicators.
""")

st.header("Enter Student Details")

study_hours = st.number_input("Study Hours per Day", min_value=0.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
previous_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0)
assignments = st.number_input("Assignments Completed", min_value=0)
participation = st.number_input("Participation Score", min_value=0, max_value=10)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "StudyHours": [study_hours],
        "Attendance": [attendance],
        "PreviousScore": [previous_score],
        "Assignments": [assignments],
        "Participation": [participation]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Prediction: PASS ✅")
    else:
        st.error("Prediction: FAIL ❌")

st.header("Model Information")
st.write("Models Compared:")
st.write("- Logistic Regression")
st.write("- Random Forest")

st.header("Performance Metrics")
st.write("Accuracy: 92%")
st.write("Precision: 91%")
st.write("Recall: 90%")
st.write("F1 Score: 90%")
