import streamlit as st
import pandas as pd
from os import path
import numpy as np
import pickle





st.title("Flower species Predictor")
petal_length = st.number_input("Please choose a petal length",
                               placeholder="Please enter value between 1 and 6.9",
                              min_value=1.0, max_value=6.9, value=None)

petal_width = st.number_input("Please choose petal width",placeholder="Please enter value between 0 and 2.5",
                              min_value=0.100000, max_value=2.500000)
speal_length = st.number_input("please choose speal length",placeholder="please enter value between 4.3 and 7.9 ",
                              min_value=4.300000, max_value=7.900000, value=None)
speal_width = st.number_input("please choose speal width",placeholder="Please enter value between 2 and 4.4",
                              min_value=2.00000, max_value=4.400000, value=None)

user_input= pd.DataFrame([[speal_length,speal_width,petal_length,petal_width]],
                         columns=['sepal_length','sepal_width','petal_length','petal_width'])

#using the .pkl file, creating an ML named 'iris_preditor'
model_path =path.join("Model", "iris_model.pkl")
with open(model_path, 'rb') as file:
    iris_predictor= pickle.load(file)

dict_species={0:'setosa', 1:'versicolor', 2:'virginica'}

if st.button("Predict species"):
         if((petal_length==None) or (petal_width==None) or (speal_length==None) or (speal_width==None)):
             st.write("please fill all values")
         else:
             #prediction can be done here. We are expecting a dataframe
             predicted_species = iris_predictor.predict(user_input)
             #predicted_species[0] will give us the value in the dataframe
             #we use that value to find the corresponding species from the
             #dictionery 'dict_species'
             st.write("the species is",dict_species[predicted_species[0]])