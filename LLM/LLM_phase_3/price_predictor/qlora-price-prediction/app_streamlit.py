import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ========================
# 🌈 Custom CSS Styling
# ========================
st.markdown("""
<style>
/* Global background */
.stApp {
    background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
    font-family: 'Poppins', sans-serif;
}

/* Glassmorphism card style */
div[data-testid="stForm"], .stTextInput > div > div, .stNumberInput > div > div, .stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 8px;
    transition: all 0.3s ease;
}
.stTextInput > div > div:hover, .stNumberInput > div > div:hover {
    transform: scale(1.01);
    border-color: rgba(255,255,255,0.6);
}

/* Title Styling */
h1 {
    color: #ffffff;
    text-align: center;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    letter-spacing: 1px;
    background: linear-gradient(90deg, #ffeaa7, #fd79a8, #a29bfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: fadeIn 2s ease-in-out;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #f1f2f6;
    margin-top: -10px;
    margin-bottom: 25px;
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    color: #4a4a4a;
    border: none;
    border-radius: 30px;
    padding: 14px 35px;
    font-weight: 600;
    font-size: 17px;
    cursor: pointer;
    box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-3px);
    background: linear-gradient(135deg, #b2fefa 0%, #e7d9ff 100%);
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}

/* Info & Success Boxes */
.stSuccess, .stInfo {
    background: rgba(255, 255, 255, 0.25);
    border-left: 6px solid #b8e6d5;
    border-radius: 12px;
    padding: 20px;
    backdrop-filter: blur(12px);
    color: #2d3436;
    font-weight: 500;
}

/* Expander customization */
.streamlit-expanderHeader {
    background-color: rgba(255,255,255,0.2) !important;
    border-radius: 10px;
    color: #2c3e50;
    font-weight: 600;
}

/* Footer */
.footer {
    text-align: center;
    color: #f5f6fa;
    font-size: 13px;
    margin-top: 40px;
    opacity: 0.8;
}

/* Fade-in animation */
@keyframes fadeIn {
  from {opacity: 0; transform: translateY(-10px);}
  to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ========================
# 🌟 App Title
# ========================
st.markdown("<h1>💰 AI Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Smart Estimation with Elegance ✨</p>", unsafe_allow_html=True)

# ========================
# 🧾 Input Section
# ========================
st.markdown("### 🧠 Enter Product Details")

col1, col2 = st.columns(2)

with col1:
    product_name = st.text_input("🏷️ Product Name", placeholder="e.g., Wireless Headphones")
    brand = st.text_input("🏢 Brand", placeholder="e.g., Sony")
    category = st.text_input("📦 Category", placeholder="e.g., Electronics")

with col2:
    color = st.text_input("🎨 Color", placeholder="e.g., Black")
    weight = st.number_input("⚖️ Weight (kg)", min_value=0.0, value=1.0, step=0.1)
    feature_score = st.number_input("⭐ Feature Score (0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# ========================
# 🤖 Model Loading
# ========================
@st.cache_resource
def load_model():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ========================
# 🔮 Predict Button
# ========================
st.markdown("<br>", unsafe_allow_html=True)
center_col = st.columns([1, 2, 1])[1]
with center_col:
    predict_button = st.button("🔮 Predict Price", use_container_width=True)

# ========================
# 📊 Prediction Logic
# ========================
if predict_button:
    if not product_name or not brand or not category:
        st.warning("⚠️ Please fill in at least Product Name, Brand, and Category.")
    else:
        with st.spinner("✨ Generating prediction... please wait"):
            prompt = f"""
Question: What is the estimated price in Indian Rupees (INR) for this product?

Product Details:
- Name: {product_name}
- Brand: {brand}
- Category: {category}
- Color: {color}
- Weight: {weight} kg
- Quality Score: {feature_score}/1.0

Answer: The estimated price is approximately INR"""
            
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.92,
                    top_k=50,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if "Answer:" in response:
                generated_text = response.split("Answer:")[-1].strip()
            else:
                generated_text = response[len(prompt):].strip()
            generated_text = generated_text.split('\n')[0].split('.')[0].strip()
            
            st.markdown("---")
            st.markdown("### 💵 Prediction Result")
            st.success(f"**{generated_text}**")
            
            with st.expander("ℹ️ More Info"):
                st.info("""
                **Model Info:**
                - Using DistilGPT2 (lightweight and CPU-friendly)
                - Text-generation model, not trained specifically on prices

                **Recommendations:**
                - Fine-tune using your product pricing dataset for accuracy
                - Include numerical and categorical embeddings
                - Apply domain-specific rules or regression models

                🧩 *This is an AI-based estimate for demonstration purposes only.*
                """)

# ========================
# 🌸 Footer
# ========================
st.markdown("<p class='footer'>© 2025 Price Predictor • Powered by 🤖 DistilGPT2 | Styled with 💖 Glassmorphism</p>", unsafe_allow_html=True)
