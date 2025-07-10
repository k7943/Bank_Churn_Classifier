import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('artifacts/model.h5')

# Load the encoders and scaler
with open('artifacts/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('artifacts/one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('artifacts/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Page config
st.set_page_config(page_title='Customer Churn Classifier', layout='centered')
st.title('Customer Churn Prediction')

st.markdown("Adjust the customer profile fields to estimate the likelihood of churn.")

# --- Manual Input Form ---
st.subheader("Customer Profile Input")

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
    age = st.slider('Age', 18, 92, 35)
    tenure = st.slider('Tenure (Years with Bank)', 0, 10, 5)

with col2:
    balance = st.number_input('Balance', min_value=0.0, value=50000.0)
    num_of_products = st.slider('Number of Products', 1, 4, 2)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])
    estimated_salary = st.number_input('Estimated Salary', value=60000.0)

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.subheader("Prediction Result")
st.metric("Churn Probability", f"{prediction_proba:.2%}")
st.progress(float(min(prediction_proba, 1.0)))

if prediction_proba > 0.5:
    st.warning('The customer is likely to churn.')
else:
    st.success('The customer is not likely to churn.')

# --- Batch Prediction Option ---
st.subheader("Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Drop irrelevant columns
        df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'], errors='ignore')

        # Encode Gender
        df['Gender'] = label_encoder_gender.transform(df['Gender'])

        # One-hot encode Geography
        geo_encoded_batch = one_hot_encoder_geo.transform(df[['Geography']]).toarray()
        geo_encoded_df_batch = pd.DataFrame(geo_encoded_batch, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
        df = pd.concat([df.drop(columns=['Geography']).reset_index(drop=True), geo_encoded_df_batch], axis=1)

        # Scale
        df_scaled = scaler.transform(df)

        # Predict
        predictions = model.predict(df_scaled).flatten()
        df['Churn Probability'] = predictions
        df['Churn Prediction'] = (predictions > 0.5).astype(int)

        st.write("Prediction Results")
        st.dataframe(df[['Churn Probability', 'Churn Prediction']])

    except Exception as e:
        st.error("Invalid or inconsistent file. Please upload a valid CSV with the correct format.")
