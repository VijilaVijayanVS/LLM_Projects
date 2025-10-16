import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ========================
# 🤖 Model Loading
# ========================
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()

# ========================
# 🔮 Prediction Function
# ========================
def predict_price(product_name, brand, category, color, weight, feature_score):
    if not product_name or not brand or not category:
        return "⚠️ Please fill in at least Product Name, Brand, and Category.", ""
    
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
    
    result = f"### 💰 Predicted Price\n\n{generated_text}"
    
    info = """
### 🤖 Model Information

**Architecture:** DistilGPT2 (Lightweight & CPU-Optimized)

**Current Limitations:**
- General-purpose text model, not price-specific trained
- Estimates are demonstrative only

**Improvement Recommendations:**
- Fine-tune on your product pricing dataset
- Incorporate numerical features and embeddings
- Consider ensemble with regression models

*This is an AI-based demonstration. Real pricing requires domain-specific training.*
"""
    
    return result, info

# ========================
# 🎨 Light Gray Theme CSS
# ========================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global styling */
.gradio-container {
    background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 50%, #f5f7fa 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Main container */
.main {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 30px 15px !important;
}

/* Header styling */
.header-title {
    text-align: center;
    font-size: 2.5em;
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #4a5568 0%, #718096 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
    animation: fadeInDown 0.8s ease-out;
}

.subtitle {
    text-align: center;
    font-size: 1em;
    color: #64748b;
    font-weight: 400;
    margin-bottom: 30px;
    letter-spacing: 0.5px;
}

/* Section headers */
.markdown h3 {
    color: #334155 !important;
    font-weight: 600 !important;
    font-size: 1em !important;
    margin-bottom: 15px !important;
    letter-spacing: 0.3px !important;
}

/* Input container styling */
.input-container {
    background: rgba(255, 255, 255, 0.8) !important;
    border-radius: 16px !important;
    padding: 25px !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06) !important;
    margin-bottom: 20px !important;
}

/* Input fields */
label {
    color: #475569 !important;
    font-weight: 500 !important;
    font-size: 0.95em !important;
    margin-bottom: 8px !important;
}

input[type="text"], input[type="number"], textarea {
    background: rgba(248, 250, 252, 0.9) !important;
    border: 1.5px solid rgba(203, 213, 225, 0.6) !important;
    border-radius: 12px !important;
    color: #334155 !important;
    padding: 12px 16px !important;
    font-size: 0.95em !important;
    transition: all 0.3s ease !important;
}

input[type="text"]:focus, input[type="number"]:focus, textarea:focus {
    background: #ffffff !important;
    border-color: #94a3b8 !important;
    box-shadow: 0 0 0 3px rgba(148, 163, 184, 0.15) !important;
    transform: translateY(-1px) !important;
}

input[type="text"]:hover, input[type="number"]:hover {
    border-color: #94a3b8 !important;
    background: #ffffff !important;
}

/* Slider styling */
input[type="range"] {
    background: transparent !important;
}

input[type="range"]::-webkit-slider-track {
    background: linear-gradient(to right, #cbd5e1, #94a3b8) !important;
    height: 6px !important;
    border-radius: 3px !important;
}

input[type="range"]::-webkit-slider-thumb {
    background: linear-gradient(135deg, #64748b, #94a3b8) !important;
    width: 18px !important;
    height: 18px !important;
    border-radius: 50% !important;
    cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
}

/* Button styling */
button {
    background: linear-gradient(135deg, #64748b 0%, #94a3b8 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 40px !important;
    font-weight: 600 !important;
    font-size: 1.05em !important;
    cursor: pointer !important;
    box-shadow: 0 4px 20px rgba(100, 116, 139, 0.25) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: 0.3px !important;
    margin-top: 10px !important;
}

button:hover {
    transform: translateY(-2px) !important;
    background: linear-gradient(135deg, #475569 0%, #64748b 100%) !important;
    box-shadow: 0 6px 25px rgba(100, 116, 139, 0.35) !important;
}

button:active {
    transform: translateY(0px) !important;
}

/* Output boxes */
.output-box {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%) !important;
    border-left: 4px solid #94a3b8 !important;
    border-radius: 16px !important;
    padding: 28px !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(203, 213, 225, 0.3) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06) !important;
    margin-top: 20px !important;
}

.output-box h3 {
    color: #1e293b !important;
    font-size: 1.4em !important;
    margin-bottom: 15px !important;
}

.output-box p, .output-box ul, .output-box li {
    color: #475569 !important;
    line-height: 1.8 !important;
    font-size: 0.98em !important;
}

.output-box strong {
    color: #334155 !important;
}

/* Accordion */
.accordion {
    background: rgba(255, 255, 255, 0.7) !important;
    border: 1px solid rgba(203, 213, 225, 0.3) !important;
    border-radius: 14px !important;
    margin-top: 20px !important;
}

.accordion summary {
    color: #475569 !important;
    font-weight: 500 !important;
    padding: 16px !important;
    cursor: pointer !important;
}

/* Divider */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(to right, transparent, rgba(203, 213, 225, 0.5), transparent) !important;
    margin: 35px 0 !important;
}

/* Footer */
.footer-text {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 0.88em !important;
    margin-top: 50px !important;
    padding-top: 30px !important;
    border-top: 1px solid rgba(203, 213, 225, 0.3) !important;
    font-weight: 400 !important;
    letter-spacing: 0.3px !important;
}

/* Animations */
@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Column styling */
.gr-column {
    animation: fadeIn 0.6s ease-out !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #f1f5f9;
}

::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}
"""

# ========================
# 🎨 Gradio Interface
# ========================
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1 class='header-title'>💰 AI Price Predictor</h1>")
    gr.HTML("<p class='subtitle'>Intelligent Product Pricing Estimation</p>")
    
    with gr.Group(elem_classes="input-container"):
        gr.Markdown("### 📋 Product Information")
        
        with gr.Row():
            with gr.Column():
                product_name = gr.Textbox(
                    label="🏷️ Product Name",
                    placeholder="e.g., Wireless Headphones",
                )
                brand = gr.Textbox(
                    label="🏢 Brand",
                    placeholder="e.g., Sony",
                )
                category = gr.Textbox(
                    label="📦 Category",
                    placeholder="e.g., Electronics",
                )
            
            with gr.Column():
                color = gr.Textbox(
                    label="🎨 Color",
                    placeholder="e.g., Black",
                )
                weight = gr.Number(
                    label="⚖️ Weight (kg)",
                    value=1.0,
                    minimum=0.0,
                    step=0.1,
                )
                feature_score = gr.Slider(
                    label="⭐ Quality Score",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                )
        
        predict_btn = gr.Button("🔮 Generate Price Estimate", size="lg")
    
    gr.Markdown("---")
    
    with gr.Row():
        result_output = gr.Markdown(elem_classes="output-box")
    
    with gr.Accordion("ℹ️ Model Details & Recommendations", open=False, elem_classes="accordion"):
        info_output = gr.Markdown(elem_classes="output-box")
    
    predict_btn.click(
        fn=predict_price,
        inputs=[product_name, brand, category, color, weight, feature_score],
        outputs=[result_output, info_output]
    )
    
    gr.HTML("<p class='footer-text'>Powered by DistilGPT2 • Built with Gradio • © 2025</p>")

# ========================
# 🚀 Launch App
# ========================
if __name__ == "__main__":
    demo.launch(share=True)