# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:17:28 2024

@author: Pavani Alla
"""

import streamlit as st
from joblib import load
import pickle
import pandas as pd
import numpy as np

# Load the dataset and extract unique values for animal names and symptoms
@st.cache_resource
def load_data(file_path='data.csv'):
    data = pd.read_csv(file_path)
    animal_names = data['AnimalName'].unique().tolist()
    # Gather unique symptoms from all symptom columns
    symptoms_columns = ['symptoms1', 'symptoms2', 'symptoms3', 'symptoms4', 'symptoms5']
    unique_symptoms = pd.unique(data[symptoms_columns].values.ravel('K')).tolist()
    return animal_names, unique_symptoms

# Load the ML model
@st.cache_data
def load_model():
    model = load('random_forest_model.joblib')
    return model

# Load the encoder used during training
@st.cache_data
def load_encoder():
    with open('onehot_encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
    return encoder

# Load unique animal names and symptoms from the dataset
animal_names, symptoms = load_data()  # Dynamically fetch animal names and symptoms

# Load model and encoder
model = load_model()
encoder = load_encoder()

# Streamlit app title and instructions
st.title("Animal Symptom Prediction App")
st.write("Select the inputs to predict whether the case is dangerous or not.")

# User input for animal name and symptoms
animal_name = st.selectbox("Select Animal Name:", options=animal_names)
symptom1 = st.selectbox("Select Symptom 1:", options=symptoms)
symptom2 = st.selectbox("Select Symptom 2:", options=symptoms)
symptom3 = st.selectbox("Select Symptom 3:", options=symptoms)
symptom4 = st.selectbox("Select Symptom 4:", options=symptoms)
symptom5 = st.selectbox("Select Symptom 5:", options=symptoms)

if st.button("Predict"):
    # Prepare the input data
    input_data = [[animal_name, symptom1, symptom2, symptom3, symptom4, symptom5]]

    # Apply one-hot encoding to the input data
    try:
        encoded_input = encoder.transform(input_data)
    except ValueError as e:
        st.error(f"Error in encoding input: {e}")
        st.stop()

    # Ensure the encoded input is in the correct format (numpy array)
    encoded_input = np.array(encoded_input)

    # Make the prediction with the model
    prediction = model.predict(encoded_input)

    # Display the result
    st.write(f"Prediction: {'Dangerous' if prediction[0] == 1 else 'Not Dangerous'}")
