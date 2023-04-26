import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

#Load the trained LSTM model
model = load_model('lstm_model.h5')

# Create a Streamlit app
st.title("LSTM Time Series Prediction")

# Collect input data from the user
input1 = st.number_input("Input 1")
input2 = st.number_input("Input 2")
input3 = st.number_input("Input 3")
input4 = st.number_input("Input 4")

# Button to trigger prediction
if st.button("Predict"):
    input_data = np.array([[[input1, input2, input3]]])
    prediction = model.predict(input_data)
    st.write("Preiction: ", prediction[0][0])