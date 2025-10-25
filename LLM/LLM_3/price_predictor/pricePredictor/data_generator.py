import pandas as pd
import numpy as np
from typing import List, Dict
import random

class PriceDataGenerator:
    """Generate synthetic product pricing data"""
    
    CATEGORIES = [
        "Electronics", "Clothing", "Home & Kitchen", "Sports",
        "Books", "Toys", "Beauty", "Automotive", "Food", "Health"
    ]
    
    BRANDS = [
        "Premium", "Standard", "Budget", "Luxury", "Generic",
        "EcoFriendly", "TechPro", "HomeStyle", "SportMax", "BasicValue"
    ]
    
    CONDITIONS = ["New", "Like New", "Good", "Fair", "Refurbished"]
    
    def __init__(self, num_samples: int = 10000):
        self.num_samples = num_samples
        np.random.seed(42)
        random.seed(42)
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate synthetic dataset with realistic pricing logic"""
        data = []
        
        for _ in range(self.num_samples):
            category = random.choice(self.CATEGORIES)
            brand = random.choice(self.BRANDS)
            condition = random.choice(self.CONDITIONS)
            
            # Base price based on category
            base_prices = {
                "Electronics": 500, "Clothing": 50, "Home & Kitchen": 80,
                "Sports": 100, "Books": 20, "Toys": 30, "Beauty": 40,
                "Automotive": 200, "Food": 15, "Health": 60
            }
            base_price = base_prices.get(category, 50)
            
            # Brand multipliers
            brand_multipliers = {
                "Luxury": 3.0, "Premium": 2.0, "Standard": 1.2,
                "Budget": 0.7, "Generic": 0.5, "EcoFriendly": 1.5,
                "TechPro": 2.2, "HomeStyle": 1.3, "SportMax": 1.8,
                "BasicValue": 0.6
            }
            
            # Condition multipliers
            condition_multipliers = {
                "New": 1.0, "Like New": 0.85, "Good": 0.7,
                "Fair": 0.5, "Refurbished": 0.65
            }
            
            age = np.random.randint(0, 60)  # months
            rating = round(np.random.uniform(3.0, 5.0), 1)
            num_reviews = np.random.randint(0, 1000)
            weight = round(np.random.uniform(0.1, 20.0), 2)
            
            # Calculate price with various factors
            price = base_price * brand_multipliers[brand] * condition_multipliers[condition]
            price *= (1 - age * 0.01)  # Depreciation
            price *= (0.8 + rating * 0.04)  # Rating bonus
            price = max(5, round(price + np.random.normal(0, price * 0.1), 2))
            
            data.append({
                "category": category,
                "brand": brand,
                "condition": condition,
                "age_months": age,
                "rating": rating,
                "num_reviews": num_reviews,
                "weight_kg": weight,
                "price": price
            })
        
        return pd.DataFrame(data)
    
    def create_instruction_dataset(self, df: pd.DataFrame) -> List[Dict]:
        """Convert to instruction format for LLM fine-tuning"""
        instructions = []
        
        templates = [
            "Predict the price for a {condition} {brand} product in {category} category, aged {age_months} months, with {rating} stars and {num_reviews} reviews.",
            "What's the estimated price of a {brand} {category} item that is {condition}, {age_months} months old, rated {rating}/5 with {num_reviews} reviews?",
            "Calculate price: Category={category}, Brand={brand}, Condition={condition}, Age={age_months}mo, Rating={rating}, Reviews={num_reviews}",
            "Price estimation needed for {condition} {category} product by {brand}, {age_months} months old, {rating} star rating, {num_reviews} customer reviews.",
        ]
        
        for _, row in df.iterrows():
            template = random.choice(templates)
            instruction = template.format(**row.to_dict())
            
            response = f"Based on the product details, the estimated price is ${row['price']:.2f}"
            
            instructions.append({
                "instruction": instruction,
                "input": "",
                "output": response,
                "price": row['price']
            })
        
        return instructions