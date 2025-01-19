import streamlit as st
from transformers import pipeline

st.title("Fine-Tuned BERT for Movie Review Classification")

classifier = pipeline('text-classification', model='distilbert-base-uncased-sentiment-model')

text = st.text_area("Enter the review here")

if st.button("Predict"):
        result = classifier(text)
        st.write("Prediction Result:", result)
