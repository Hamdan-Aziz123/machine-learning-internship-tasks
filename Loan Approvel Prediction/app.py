import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('loan_approval_model.pkl', 'rb'))
st.title("Loan Approval Prediction App")
st.write("Enter the details below to predict loan approval status:")


    #no_of_dependents	education	self_employed	income_annum	loan_amount	loan_term
st.subheader("User Input Features")
st.slider("Number of Dependents", 0, 10, 2, 1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.slider("Annual Income", 0, 10000000, 9600000, 1000)
loan_amount = st.slider("Loan Amount", 0, 100000000, 29900000, 1000)
loan_term = st.slider("Loan Duration (Months)", 0, 360, 12, 1)

if education == "Graduate":
    education = 1
else:
    education = 0

if self_employed == "Yes":
    self_employed = 1
else:
    self_employed = 0

input_data = np.array([[0, education, self_employed, income_annum, loan_amount, loan_term]])

if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
    


    



