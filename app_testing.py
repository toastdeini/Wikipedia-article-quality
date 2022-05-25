# Imports
import streamlit as st
import pickle
# import parse_it

f = open('models/xgb_model.sav', 'rb')
final_model = pickle.load(f)

decoder = {
    0 : 'Good article!',
    1 : 'Not looking so great...'
}

st.title("Is this Wikipedia article up to snuff?")