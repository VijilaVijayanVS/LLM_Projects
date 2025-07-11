# app.py
import streamlit as st
from text_summarizer import TextSummarizer
import random

st.set_page_config(page_title="AI Chatbot Summarizer", layout="wide")
st.title("🤖 AI Chatbot with File Summarization")

summarizer = TextSummarizer()

uploaded_file = st.file_uploader("📂 Upload a text or markdown file", type=["txt", "md"])

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    summary = summarizer.summarize_text(content)

    # Display original text
    st.subheader("📄 Original Text")
    with st.expander("Show full text"):
        st.write(content)

    # Display statistics
    st.subheader("📊 Text Statistics")
    st.write(summary['statistics'])

    # Key phrases
    st.subheader("🔑 Key Phrases")
    for phrase, count in summary['key_phrases']:
        st.markdown(f"- **{phrase}** ({count} times)")

    # Summary
    st.subheader("📝 Summary")
    st.write(summary['summary'])

    # Compression
    st.subheader("📉 Compression Ratio")
    ratio = (1 - summary['summary_length'] / summary['original_length']) * 100
    st.write(f"Original Length: {summary['original_length']} characters")
    st.write(f"Summary Length: {summary['summary_length']} characters")
    st.write(f"Compression: {ratio:.2f}%")

    # Simple chatbot
    st.subheader("💬 Chat with the File")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask anything about the content...")
    if user_input:
        # VERY BASIC Chatbot (keyword matching from summary)
        matched_phrases = [phrase for phrase, _ in summary['key_phrases'] if phrase in user_input.lower()]
        if matched_phrases:
            response = f"The file discusses topics like: {', '.join(matched_phrases)}"
        else:
            response = random.choice([
                "That's interesting! This file seems to focus on key concepts discussed in the summary.",
                "Hmm, that topic isn't clearly mentioned, but the summary might help!",
                "I didn't find that exactly, but here's something from the summary."
            ])

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")
