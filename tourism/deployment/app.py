import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="divyavarmamisc/prod_taken_model", filename="best_prod_taken_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Product Taken Prediction
st.title("Product Taken Prediction App")
st.write("""
This application predicts whether a customer will take the pitched product based on their demographic and interaction data.
Please enter the customer details below to get a prediction.
""")

# User input fields based on the features used in the model
st.header("Customer Information")

# Numeric features
age = st.number_input("Age", min_value=18.0, max_value=100.0, value=30.0, step=1.0)
duration_of_pitch = st.number_input("Duration Of Pitch (minutes)", min_value=0.0, max_value=60.0, value=10.0, step=1.0)
number_of_person_visiting = st.number_input("Number Of Person Visiting", min_value=1, max_value=10, value=2, step=1)
number_of_followups = st.number_input("Number Of Followups", min_value=0.0, max_value=10.0, value=4.0, step=1.0)
preferred_property_star = st.selectbox("Preferred Property Star", options=[1.0, 2.0, 3.0, 4.0, 5.0], index=2)
number_of_trips = st.number_input("Number Of Trips", min_value=1.0, max_value=50.0, value=5.0, step=1.0)
number_of_children_visiting = st.number_input("Number Of Children Visiting", min_value=0.0, max_value=5.0, value=1.0, step=1.0)
monthly_income = st.number_input("Monthly Income", min_value=0.0, max_value=100000.0, value=20000.0, step=1000.0)

# Categorical features
type_of_contact = st.selectbox("Type of Contact", options=['Self Enquiry', 'Company Invited'], index=0)
city_tier = st.selectbox("City Tier", options=[1, 2, 3], index=0)
occupation = st.selectbox("Occupation", options=['Salaried', 'Small Business', 'Large Business', 'Unemployed'], index=0)
gender = st.selectbox("Gender", options=['Male', 'Female'], index=0)
product_pitched = st.selectbox("Product Pitched", options=['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King'], index=1)
marital_status = st.selectbox("Marital Status", options=['Married', 'Single', 'Divorced'], index=0)
passport = st.selectbox("Passport", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', index=0)
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", options=[1, 2, 3, 4, 5], index=2)
own_car = st.selectbox("Own Car", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', index=0)
designation = st.selectbox("Designation", options=['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'], index=0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])


if st.button("Predict if Product is Taken"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        result = "Customer WILL take the product!"
        st.success(f"The model predicts: **{result}**")
    else:
        result = "Customer will NOT take the product."
        st.warning(f"The model predicts: **{result}**")
