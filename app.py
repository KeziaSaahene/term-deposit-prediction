import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("term_deposit_model.pkl")



emp_var_rate = st.number_input("Employment Variation Rate (emp.var.rate)", value=1.0)
cons_price_idx = st.number_input("Consumer Price Index (cons.price.idx)", value=93.2)
cons_conf_idx = st.number_input("Consumer Confidence Index (cons.conf.idx)", value=-40.0)
euribor3m = st.number_input("3-Month Euribor Rate (euribor3m)", value=0.5)
nr_employed = st.number_input("Number of Employees (nr.employed)", value=5000.0)

# Mappings (these must match the ones used during model training)
job_map = {'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3,
           'management': 4, 'retired': 5, 'self-employed': 6, 'services': 7,
           'student': 8, 'technician': 9, 'unemployed': 10, 'unknown': 11}

education_map = {'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2, 'high.school': 3,
                 'illiterate': 4, 'professional.course': 5, 'university.degree': 6,
                 'unknown': 7}

default_map = {'no': 0, 'yes': 2, 'unknown': 1}
housing_map = {'no': 0, 'yes': 2, 'unknown': 1}
contact_map = {'cellular': 0, 'telephone': 1}
month_map = {'apr': 0, 'aug': 1, 'dec': 2, 'jul': 3, 'jun': 4, 'mar': 5,
             'may': 6, 'nov': 7, 'oct': 8, 'sep': 9}
day_map = {'mon': 1, 'tue': 3, 'wed': 4, 'thu': 2, 'fri': 0}
poutcome_map = {'failure': 0, 'nonexistent': 1, 'success': 2}

# App UI
st.title("Term Deposit Subscription Predictor")

age = st.slider("Age", 18, 95)
job = st.selectbox("Job", list(job_map.keys()))
education = st.selectbox("Education", list(education_map.keys()))
default = st.selectbox("Credit Default", list(default_map.keys()))
housing = st.selectbox("Housing Loan", list(housing_map.keys()))
contact = st.selectbox("Contact Communication", list(contact_map.keys()))
month = st.selectbox("Last Contact Month", list(month_map.keys()))
day_of_week = st.selectbox("Day of Week", list(day_map.keys()))
poutcome = st.selectbox("Previous Campaign Outcome", list(poutcome_map.keys()))
duration = st.number_input("Last Contact Duration (seconds)", 0, 5000, 100)
campaign = st.number_input("Number of Contacts During Campaign", 1, 50, 1)
pdays = st.number_input("Days Since Last Contact", 0, 999, 999)
previous = st.number_input("Number of Previous Contacts", 0, 50, 0)


# Internally calculate age group (not shown on app)
def get_age_group(age):
    if age < 26:
        return "18-25"
    elif age < 36:
        return "26-35"
    elif age < 46:
        return "36-45"
    elif age < 56:
        return "46-55"
    elif age < 66:
        return "56-65"
    else:
        return "66+"

age_group_map = {
    "18-25": 0,
    "26-35": 1,
    "36-45": 2,
    "46-55": 3,
    "56-65": 4,
    "66+": 5
}
age_group = get_age_group(age)
age_group_encoded = age_group_map[age_group]

# Prediction button
if st.button("Predict"):
    # Encode categorical variables
    features = np.array([
        age,
        duration,
        campaign,
        pdays,
        previous,
        job_map[job],
        education_map[education],
        default_map[default],
        housing_map[housing],
        contact_map[contact],
        month_map[month],
        day_map[day_of_week],
        poutcome_map[poutcome],
        emp_var_rate,
        cons_price_idx,
        cons_conf_idx,
        euribor3m,
        nr_employed, age_group_encoded  
    ])


    # Make prediction
    prediction = model.predict(features.reshape(1, -1))
    result = "Subscribed ✅" if prediction[0] == 1 else "Not Subscribed"
    st.success(f"Prediction: {result}")

