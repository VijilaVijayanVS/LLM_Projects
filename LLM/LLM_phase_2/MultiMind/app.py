import streamlit as st
from transformers import pipeline
from PIL import Image
import torch

# -------------------------------
# 🖥️ Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="🧠✨  Multi Mind", 
    page_icon="🧠✨ ", 
    layout="wide"
)

# -------------------------------
# 🎨 Custom Pastel Theme
# -------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #f3e8ff, #d0f0ea, #fff1e6); /* Lavender → Mint → Peach */
    font-family: 'Segoe UI', sans-serif;
    color: #333333;
}

/* Rounded inputs and buttons */
.stTextInput, .stFileUploader, .stButton button, .stTextArea textarea {
    border-radius: 10px;
    padding: 10px;
}

/* Button styling */
.stButton button {
    background-color: #a5c9ca;
    color: white;
    font-weight: 600;
    border: none;
    transition: 0.3s;
}
.stButton button:hover {
    background-color: #84b6b7;
}

/* Response box styling */
.response-box {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 15px;
    margin-top: 20px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("🧠✨  Multi Mind")
st.write("Ask questions based on **text or image** — powered by **Hugging Face LLMs**")

# -------------------------------
# 📦 Lazy Model Loaders (Cached)
# -------------------------------
@st.cache_resource
def get_text_model():
    return pipeline("text2text-generation", model="google/flan-t5-base", device=0 if torch.cuda.is_available() else -1)

@st.cache_resource
def get_image_caption_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=0 if torch.cuda.is_available() else -1)

@st.cache_resource
def get_vqa_model():
    return pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa", device=0 if torch.cuda.is_available() else -1)

# -------------------------------
# 🧠 UI Layout
# -------------------------------
tab1, tab2 = st.tabs(["💬 Text Assistant", "🖼️ Image Assistant"])

# ---- TEXT MODE ----
with tab1:
    uploaded_text_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])
    user_input = st.text_area("Or enter your question/prompt manually:", height=120)

    # If file is uploaded, read content
    if uploaded_text_file is not None:
        file_content = uploaded_text_file.read().decode("utf-8")
        st.info("📄 Content loaded from file.")
    else:
        file_content = ""

    if st.button("Generate Response", key="text_btn"):
        # Use file content if available, else user input
        prompt = file_content if file_content else user_input
        if prompt.strip():
            with st.spinner("Thinking... 💭"):
                model = get_text_model()
                response = model(prompt, max_length=200, do_sample=True)[0]['generated_text']
            st.markdown(f"<div class='response-box'><b>🪄 Response:</b><br>{response}</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a text prompt or upload a file.")

# ---- IMAGE MODE ----
with tab2:
    uploaded_img = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    img_question = st.text_input("Ask a question about the image (optional):", "")

    if uploaded_img:
        image = Image.open(uploaded_img)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image", key="img_btn"):
            with st.spinner("Analyzing image... 🧠"):
                if img_question.strip():
                    vqa = get_vqa_model()
                    answer = vqa(image=image, question=img_question)[0]['answer']
                    st.markdown(f"<div class='response-box'><b>🪄 Answer:</b><br>{answer}</div>", unsafe_allow_html=True)
                else:
                    captioner = get_image_caption_model()
                    caption = captioner(image)[0]['generated_text']
                    st.markdown(f"<div class='response-box'><b>🪄 Caption:</b><br>{caption}</div>", unsafe_allow_html=True)
