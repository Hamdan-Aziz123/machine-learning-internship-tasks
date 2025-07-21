import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

model = joblib.load('exam_score_predictor_model.pkl')

st.title("Customer Exam Score Predictor")


study_hours_per_day = st.slider('Study Hours per Day', 0, 12, 4)

input_data = np.array([[study_hours_per_day]])

if st.button('Predict Exam Score'):
    input_data = np.array([[study_hours_per_day]])
    prediction = model.predict(input_data)[0]
    prediction=max(0, min(100, prediction))  # Ensure the score is between 0 and 100
    st.success(f'Predicted Exam Score: {prediction:.2f}')


#now again model prediction with more features with study hours


modelAgain = joblib.load('exam_score_predictor_model_v2.pkl')

st.title("Customer Exam Score Predictor with More Features")

study_hours_per_day = st.slider('Study Hours per Day', 0.0, 12.0, 2.0)
attendance_percentage = st.slider('Attendance Percentage', 0.0, 100.0, 80.0)
mental_health_score = st.slider('Mental Health Score', 0, 10, 5)
sleep_hours_per_night = st.slider('Sleep Hours per Night', 0.0, 24.0, 7.0)
part_time_job = st.selectbox('Part Time Job', ['Yes', 'No'])
part_time_job = 1 if part_time_job == 'Yes' else 0

input_data = np.array([[study_hours_per_day, attendance_percentage, mental_health_score, sleep_hours_per_night, part_time_job]])

if st.button('Predict Exam Score with More Features'):
    input_data = np.array([[study_hours_per_day, attendance_percentage, mental_health_score, sleep_hours_per_night, part_time_job]])
    prediction = modelAgain.predict(input_data)[0]
    prediction = max(0, min(100, prediction))  # Ensure the score is between 0 and 100
    st.success(f'Predicted Exam Score with More Features: {prediction:.2f}')
