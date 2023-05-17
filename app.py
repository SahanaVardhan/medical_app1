import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
st.title("Medical Diagnostic Prediction App")
st.markdown('Does the person have Diabetes?')

## step1: Load the trained model
model=open('rfc.pickle','rb')
clf=pickle.load(model)
model.close()

## step 2 : Get the user input from the front end
pregs=st.number_input('Pregnancies',0,20,step=1)
glucose=st.slider('Glucose',40,200,40)
bp=st.slider('BloodPressure',20,140,20)
skin=st.slider('SkinThickness',7,99,7)
insulin=st.slider('Insulin',14,850,14)
bmi=st.slider('BMI',18,70,15)
diabetes=st.slider('DiabetesPedigreeFunction',0.05,2.50,0.05)
age=st.slider('Age',21,90,21)

## Step 3: Convert user input to model input
diction={'Pregnancies':pregs,'Glucose':glucose, 'BloodPressure':bp, 'SkinThickness':skin, 'Insulin':insulin,
       'BMI':bmi, 'DiabetesPedigreeFunction':diabetes, 'Age':age}
input_data=pd.DataFrame([diction])

## Step 4: Get the predictions and print the results
prediction=clf.predict(input_data)[0]
if st.button('predict'):
    if prediction==0:
        st.write('The person is healthy')
    if prediction==1:
        st.write('The person has diabetes')
