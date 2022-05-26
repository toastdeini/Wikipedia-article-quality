# Imports
import streamlit as st
import pickle
from src.parse_it import *

f = open('models/xgb_model.sav', 'rb')
model = pickle.load(f)

decoder = {
    0 : 'Good article!',
    1 : 'Not looking so great...'
}

st.title("Is this Wikipedia article up to snuff?")

# article = st.text_area("Paste the text of the article you want to analyze here!", max_chars=100000)

# pred = model.predict(parse_doc(article))