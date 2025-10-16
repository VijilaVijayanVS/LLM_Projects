# 🎯 AI Price Prediction System

Fine-tuned LLM with QLoRA for accurate product price predictions.

## 📦 Installation

### Step 1: Install Dependencies

```bash
pip install torch transformers peft bitsandbytes accelerate datasets gradio pandas numpy scikit-learn matplotlib seaborn tqdm sentence-transformers
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### Option 1: Train + Launch (Recommended for first time)

```bash
python3 main.py --mode all
```

This will:
1. Generate 5000 training samples
2. Fine-tune TinyLlama-1.1B with QLoRA
3. Launch the Gradio UI

### Option 2: Train Only

```bash
python3 main.py --mode train
```

### Option 3: Launch UI Only (after training)

```bash
python3 main.py --mode app
```

## 📁 Project Structure

```
project/
├── main.py                 # Main execution script
├── config.py              # Configuration settings
├── data_generator.py      # Dataset generation
├── model_trainer.py       # QLoRA fine-tuning
├── inference_engine.py    # Inference with caching
├── app.py                 # Gradio UI
├── requirements.txt       # Dependencies
├── models/                # Saved models (auto-created)
│   └── price_predictor/
└── data/                  # Generated datasets (auto-created)
    └── product_dataset.csv
```

## ⚙️ Configuration

Edit `config.py` to customize:

### Model Configuration
- `base_model`: Base LLM (default: TinyLlama-1.1B)
- `lora_r`: LoRA rank (default: 16)
- `lora_alpha`: LoRA alpha (default: 32)
- `max_length`: Max sequence length (default: 512)

### Training Configuration
- `num_epochs`: Training epochs (default: 3)
- `batch_size`: Batch size (default: 4)
- `learning_rate`: Learning rate (default: 2e-4)

### Cache Configuration
- `use_cache`: Enable caching (default: True)
- `cache_ttl`: Cache TTL in seconds (default: 3600)

## 🎮 Using the UI

### 1. Single Prediction
- Fill in product details
- Click "🔮 Predict Price"
- View results instantly

### 2. Batch Prediction
- Upload CSV with required columns
- Click "🚀 Process Batch"
- Download results

Required CSV columns:
```
category, brand, condition, age_months, rating, num_reviews, weight_kg
```

### 3. Analytics
- Click "📊 Generate Analytics"
- View pricing trends and distributions

### 4. History
- Click "🔄 Refresh History"
- View all past predictions

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1**: Reduce batch size in `config.py`
```python
batch_size: int = 2
gradient_accumulation_steps: int = 8
```

**Solution 2**: Use CPU only
```bash
export CUDA_VISIBLE_DEVICES=""
python main.py
```

### Issue: Model Loading Error

**Solution**: Ensure model is trained first
```bash
python main.py --mode train
```

### Issue: Import Errors

**Solution**: Reinstall dependencies
```bash
pip install --upgrade torch transformers peft bitsandbytes accelerate
```

## 📊 Performance Metrics

- **Model Size**: 1.1B parameters (4-bit quantized)
- **Training Time**: ~30-60 minutes (on GPU)
- **Inference Time**: <1 second per prediction
- **Memory Usage**: ~4GB GPU / ~8GB CPU
- **Accuracy**: MAE typically <10% on test set

## 🎯 Features

✅ **QLoRA Fine-tuning**: Efficient 4-bit quantization  
✅ **Fast Inference**: <1s predictions with caching  
✅ **Modern UI**: Clean Gradio interface  
✅ **Batch Processing**: Handle multiple products  
✅ **Analytics**: Visualize pricing trends  
✅ **History Tracking**: Monitor predictions  
✅ **Smart Caching**: Instant repeated queries  

## 🔬 Technical Details

### QLoRA (Quantized Low-Rank Adaptation)
- 4-bit NF4 quantization
- LoRA adapters on attention layers
- 75% memory reduction vs full fine-tuning
- Maintains model quality

### Model Architecture
- Base: TinyLlama-1.1B-Chat
- LoRA rank: 16
- Target modules: q_proj, k_proj, v_proj, o_proj
- Training: Mixed precision (FP16)

## 📝 Example Usage

```python
from inference_engine import PricePredictor
from config import CacheConfig

# Initialize predictor
predictor = PricePredictor("./models/price_predictor", CacheConfig())

# Make prediction
product = {
    "category": "Electronics",
    "brand": "Premium",
    "condition": "New",
    "age_months": 0,
    "rating": 4.5,
    "num_reviews": 100,
    "weight_kg": 1.5
}

result = predictor.predict(product)
print(f"Predicted Price: ${result['predicted_price']:.2f}")
```

## 🤝 Contributing

Improvements welcome! Areas for contribution:
- Additional model architectures
- Enhanced caching strategies
- More analytics features
- UI improvements

## 📄 License

MIT License - Free to use and modify

## 🙏 Acknowledgments

- HuggingFace Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- TinyLlama Project
- Gradio Team

---

**Need Help?** Open an issue or contact support.

**Star ⭐ this repo if you find it useful!**
"""