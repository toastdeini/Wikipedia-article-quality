# Imports
import streamlit as st
import pickle
from src.parse_it import *

f = open('models/xgb_model.sav', 'rb')
model = pickle.load(f)
f.close()

decoder = {
    0 : 'Looking good!',
    1 : 'Hm, this could use some work...'
}

st.title("Is this Wikipedia article up to snuff?")

article = st.text_area(
"Find out by pasting the text of your article below!",
"This is a sample text entry. Input your desired text here - the model's prediction \
will be returned below.",
height = 200,
max_chars=100000,
placeholder = 'Enter some text!'
)

pred = model.predict([parse_doc(article)])

st.write(decoder[pred[0]])