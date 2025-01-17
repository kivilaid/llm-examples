import streamlit as st
from langchain.llms import OpenAI

st.title("🦜🔗 Langchain Quickstart App")

# Use the Streamlit Secrets functionality to securely store the API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))


with st.form("my_form"):
    text = st.text_area("Enter text:", "What are 3 key advice for learning how to code?")
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
