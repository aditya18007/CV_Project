from pyparsing import opAssoc
from regex import P
import streamlit as st 
from PIL import Image

st.title("Upload + Classification Example")

def pretty_string(techniques):
    pass 

def predict(file_path):
    return ['aditya', 'rathore']

file_path = st.file_uploader("Choose an image...", type="jpg")
if file_path is not None:
    image = Image.open(file_path)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detected Propaganda techniques:")
    labels = predict(file_path)
    st.write(str(labels))