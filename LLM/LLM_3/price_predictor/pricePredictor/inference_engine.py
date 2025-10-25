import torch
import re
from typing import Optional, Dict
import hashlib
import json
from config import CacheConfig

class CacheManager:
    """Simple in-memory cache for predictions"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = {} if config.use_cache else None
        
    def get(self, key: str) -> Optional[str]:
        if not self.config.use_cache:
            return None
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        if self.config.use_cache:
            self.cache[key] = value
    
    def clear(self):
        if self.cache:
            self.cache.clear()
            print("Cache cleared!")
    
    def size(self):
        return len(self.cache) if self.cache else 0

class PricePredictor:
    """Inference engine with caching"""
    
    def __init__(self, model_path: str, cache_config: CacheConfig):
        self.model_path = model_path
        self.cache = CacheManager(cache_config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
    
    def load_model(self):
        """Load fine-tuned model for inference"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel, PeftConfig
        
        print(f"Loading model from {self.model_path}...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load PEFT config to get base model
            peft_config = PeftConfig.from_pretrained(self.model_path)
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            
            # Load PEFT model
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()
            
            print(f"✓ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _create_cache_key(self, product_details: Dict) -> str:
        """Create hash key for caching"""
        content = json.dumps(product_details, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def predict(self, product_details: Dict) -> Dict:
        """Predict price with caching support"""
        cache_key = self._create_cache_key(product_details)
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result:
            result = json.loads(cached_result)
            result["from_cache"] = True
            return result
        
        # Create prompt
        prompt = f"""<|system|>
You are a price prediction expert. Analyze product details and provide accurate price estimates.
<|user|>
Predict the price for a {product_details['condition']} {product_details['brand']} product in {product_details['category']} category, aged {product_details['age_months']} months, with {product_details['rating']} stars and {product_details['num_reviews']} reviews.
<|assistant|>
"""
        
        # Generate prediction
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("<|assistant|>")[-1].strip()
        
        # Extract price from response
        price_match = re.search(r'\$?(\d+\.?\d*)', response)
        predicted_price = float(price_match.group(1)) if price_match else 0.0
        
        result = {
            "predicted_price": predicted_price,
            "raw_response": response,
            "from_cache": False,
            "cache_size": self.cache.size()
        }
        
        # Cache the result
        self.cache.set(cache_key, json.dumps(result))
        
        return result
    
    def batch_predict(self, products: list) -> list:
        """Batch prediction for multiple products"""
        results = []
        for product in products:
            result = self.predict(product)
            results.append(result)
        return results