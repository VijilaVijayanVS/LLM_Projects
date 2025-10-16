import streamlit as st
import requests

st.set_page_config(page_title="Welcome To LLM Fusion", layout="centered")
st.title("🤖 Welcome To LLM Fusion")

option = st.radio("Choose a feature", ["Chat", "Summarize Document"])

if option == "Chat":
    user_input = st.text_input("Enter your message:")
    if st.button("Send"):
        response = requests.post("http://localhost:8000/chat", json={"message": user_input})
        st.write("**Assistant:**", response.json()["response"])

elif option == "Summarize Document":
    uploaded_file = st.file_uploader("Upload a PDF file")
    if uploaded_file and st.button("Summarize"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/summarize", files=files)
        st.subheader("Summary")
        st.write(response.json()["summary"])