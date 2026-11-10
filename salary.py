import streamlit as st
import pandas as pd
import pickle

with open("salary_data.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Salary Prediction")

st.title("Salary Prediction")
st.write("Predict your salary based on years of experience.")

experience = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=40.0,
    value=1.0,
    step=0.5
)

if st.button("Predict Salary"):

    input_data = pd.DataFrame({
        "YearsExperience": [experience]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Salary: {prediction[0]:,.2f}")

st.subheader(" Experience vs Predicted Salary")

experience_range = pd.DataFrame({
    "YearsExperience": range(1, 11)
})

salary_predictions = model.predict(experience_range)

chart_data = pd.DataFrame({
    "Years of Experience": range(1, 11),
    "Predicted Salary": salary_predictions
})

st.line_chart(
    chart_data,
    x="Years of Experience",
    y="Predicted Salary"
)