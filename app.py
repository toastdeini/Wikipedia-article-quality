# Imports
import streamlit as st
import pickle
# Source file contains both parsing function
# and default string for app on loadup
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
value=test_article_string,
height=380,
max_chars=100000,
placeholder='Enter some text!'
)

pred = model.predict([parse_doc(article)])

run = st.button("Click to run!")

if run:    

    st.write(decoder[pred[0]])