import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle
import numpy as np
import tensorflow as tf


# Load the trained model
model = tf.keras.models.load_model('ann_model.h5')

# load the encoders and scaler

with open(r'C:\Study-Material\ANN Classification\label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(r'C:\Study-Material\ANN Classification\onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open(r'C:\Study-Material\ANN Classification\scaler.pickle.pkl', 'rb') as file:
    scaler = pickle.load(file)

# streamlit app
import streamlit as st
st.title("Customer Churn Prediction")

#User inputs
Geography = st.selectbox("Geography",onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Memeber',[0,1])

input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary],
    'EstimatedSalary':[estimated_salary],
})

geo_encoder = onehot_encoder_geo.transform([[Geography]]).toarray()
geo_encoder_df = pd.DataFrame(geo_encoder,columns = onehot_encoder_geo.get_feature_names_out(['Geography']))

input_df = pd.concat([input_data.reset_index(drop=True),geo_encoder_df],axis=1)

#Scale the input data
input_scaled = scaler.transform(input_df)

#Predict Churn
prediction = model.predict(input_scaled)
predict_probs = prediction[0][0]

st.write(f"Churn Probability: {predict_probs:.2f}")

if predict_probs>0.5:
    st.write("The customer is likely to leave the bank")
else:
    st.write("The customer is likely to stay with the bank")