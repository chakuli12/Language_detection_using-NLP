# function for predicting language


#import this libries 

import pickle                #Pickle is a Python module that converts Python objects into byte streams, which can then be stored in files or databases. 
import string                # #strings in Python are arrays of bytes representing unicode characters.
import streamlit as st    #Streamlit is an open-source Python library that makes it easy to create and share custom web apps for machine learning and data science.
import webbrowser  

#webbrowser module is a convenient web browser controller. It provides a high-level interface that allows displaying Web-based documents to users. 

global Lrdetect_model
LrdetectFile=open('model.pckl','rb')
Lrdetect_model=pickle.load(LrdetectFile)
LrdetectFile.close()

st.title('Language Detection Tool')
input_test=st.text_input("Provide your text input here","Hello My name is Pooja ")

button_clicked=st.button("Get Language Name")  
if button_clicked:
  st.text(Lrdetect_model.predict([input_test]))
