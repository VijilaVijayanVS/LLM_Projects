import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from data_generator import PriceDataGenerator
from inference_engine import PricePredictor
import io

class PricePredictionApp:
    """Gradio UI for price prediction system"""
    
    def __init__(self, predictor: PricePredictor):
        self.predictor = predictor
        self.history = []
    
    def predict_price(self, category, brand, condition, age_months, 
                     rating, num_reviews, weight_kg):
        """Main prediction function"""
        try:
            product_details = {
                "category": category,
                "brand": brand,
                "condition": condition,
                "age_months": int(age_months),
                "rating": float(rating),
                "num_reviews": int(num_reviews),
                "weight_kg": float(weight_kg)
            }
            
            result = self.predictor.predict(product_details)
            
            # Add to history
            self.history.append({
                **product_details,
                "predicted_price": result["predicted_price"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            cache_msg = "✅ From Cache (Instant)" if result.get("from_cache") else "🔥 Fresh Prediction"
            cache_msg += f" | Cache Size: {result.get('cache_size', 0)}"
            
            return (
                f"💰 ${result['predicted_price']:.2f}",
                result["raw_response"],
                cache_msg
            )
        except Exception as e:
            return f"❌ Error: {str(e)}", "", ""
    
    def get_history_df(self):
        """Get prediction history as DataFrame"""
        if not self.history:
            return pd.DataFrame({"Message": ["No predictions yet. Start predicting!"]})
        return pd.DataFrame(self.history)
    
    def clear_history(self):
        """Clear prediction history"""
        self.history = []
        return pd.DataFrame({"Message": ["History cleared!"]})
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.predictor.cache.clear()
        return "🗑️ Cache cleared successfully!"
    
    def process_batch(self, file):
        """Process batch CSV file"""
        if file is None:
            return pd.DataFrame({"Error": ["Please upload a CSV file"]})
        
        try:
            df = pd.read_csv(file.name)
            required_cols = ["category", "brand", "condition", "age_months", 
                           "rating", "num_reviews", "weight_kg"]
            
            if not all(col in df.columns for col in required_cols):
                return pd.DataFrame({"Error": [f"CSV must contain columns: {required_cols}"]})
            
            products = df.to_dict('records')
            results = self.predictor.batch_predict(products)
            
            df['predicted_price'] = [r['predicted_price'] for r in results]
            df['from_cache'] = [r['from_cache'] for r in results]
            
            return df
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})
    
    def generate_analytics(self):
        """Generate analytics visualization"""
        if not self.history:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data yet. Make some predictions!', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        df = pd.DataFrame(self.history)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Price Prediction Analytics', fontsize=16, fontweight='bold')
        
        # Price by category
        df.groupby('category')['predicted_price'].mean().sort_values().plot(
            kind='barh', ax=axes[0, 0], color='skyblue'
        )
        axes[0, 0].set_title('Average Price by Category')
        axes[0, 0].set_xlabel('Price ($)')
        
        # Price by brand
        df.groupby('brand')['predicted_price'].mean().sort_values().plot(
            kind='barh', ax=axes[0, 1], color='lightcoral'
        )
        axes[0, 1].set_title('Average Price by Brand')
        axes[0, 1].set_xlabel('Price ($)')
        
        # Price distribution
        axes[1, 0].hist(df['predicted_price'], bins=20, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Price Distribution')
        axes[1, 0].set_xlabel('Price ($)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Price by condition
        df.groupby('condition')['predicted_price'].mean().sort_values().plot(
            kind='bar', ax=axes[1, 1], color='plum'
        )
        axes[1, 1].set_title('Average Price by Condition')
        axes[1, 1].set_xlabel('Condition')
        axes[1, 1].set_ylabel('Price ($)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_ui(self):
        """Create Gradio interface"""
        
        custom_css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .primary-btn {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        }
        """
        
        with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="🎯 AI Price Predictor") as demo:
            
            gr.Markdown("""
            # 🎯 AI-Powered Price Prediction System
            ### Fine-tuned LLM with QLoRA | Fast • Accurate • Intelligent
            
            This system uses a fine-tuned **TinyLlama-1.1B** model with **QLoRA (4-bit quantization)** 
            to predict product prices based on multiple features.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Product Details")
                    
                    category = gr.Dropdown(
                        choices=PriceDataGenerator.CATEGORIES,
                        label="📦 Category",
                        value="Electronics",
                        info="Select product category"
                    )
                    
                    brand = gr.Dropdown(
                        choices=PriceDataGenerator.BRANDS,
                        label="🏷️ Brand",
                        value="Standard",
                        info="Select brand type"
                    )
                    
                    condition = gr.Dropdown(
                        choices=PriceDataGenerator.CONDITIONS,
                        label="✨ Condition",
                        value="New",
                        info="Product condition"
                    )
                    
                    with gr.Row():
                        age_months = gr.Slider(
                            0, 60, value=0, step=1,
                            label="📅 Age (months)",
                            info="Product age"
                        )
                        rating = gr.Slider(
                            1.0, 5.0, value=4.5, step=0.1,
                            label="⭐ Rating",
                            info="Customer rating"
                        )
                    
                    with gr.Row():
                        num_reviews = gr.Number(
                            value=100,
                            label="💬 # Reviews",
                            info="Number of reviews"
                        )
                        weight_kg = gr.Number(
                            value=1.0,
                            label="⚖️ Weight (kg)",
                            info="Product weight"
                        )
                    
                    with gr.Row():
                        predict_btn = gr.Button(
                            "🔮 Predict Price",
                            variant="primary",
                            size="lg",
                            elem_classes=["primary-btn"]
                        )
                        clear_btn = gr.Button("🗑️ Clear", size="sm")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 💰 Prediction Results")
                    
                    price_output = gr.Textbox(
                        label="Predicted Price",
                        lines=2,
                        interactive=False,
                        placeholder="Click 'Predict Price' to get results..."
                    )
                    
                    cache_status = gr.Textbox(
                        label="Cache Status",
                        lines=1,
                        interactive=False
                    )
                    
                    response_output = gr.Textbox(
                        label="Model Response",
                        lines=5,
                        interactive=False,
                        placeholder="Full model response will appear here..."
                    )
                    
                    clear_cache_btn = gr.Button("🗑️ Clear Cache", size="sm")
                    cache_msg = gr.Textbox(label="Cache Info", lines=1, interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("## 🚀 Advanced Features")
            
            with gr.Tabs():
                with gr.Tab("📊 Batch Prediction"):
                    gr.Markdown("""
                    ### Upload CSV for Batch Processing
                    Required columns: `category`, `brand`, `condition`, `age_months`, `rating`, `num_reviews`, `weight_kg`
                    """)
                    
                    batch_file = gr.File(
                        label="Upload CSV File",
                        file_types=[".csv"]
                    )
                    batch_btn = gr.Button("🚀 Process Batch", variant="primary")
                    batch_output = gr.Dataframe(
                        label="Batch Results",
                        interactive=False
                    )
                    
                    gr.Markdown("""
                    **Example CSV format:**
                    ```
                    category,brand,condition,age_months,rating,num_reviews,weight_kg
                    Electronics,Premium,New,0,4.5,100,1.5
                    Clothing,Budget,Good,12,4.0,50,0.5
                    ```
                    """)
                
                with gr.Tab("📈 Analytics Dashboard"):
                    gr.Markdown("### Visualization of Prediction Trends")
                    analytics_btn = gr.Button("📊 Generate Analytics", variant="primary")
                    analytics_plot = gr.Plot(label="Analytics Visualizations")
                
                with gr.Tab("🕒 Prediction History"):
                    gr.Markdown("### View All Past Predictions")
                    with gr.Row():
                        history_btn = gr.Button("🔄 Refresh History", variant="primary")
                        clear_hist_btn = gr.Button("🗑️ Clear History", variant="stop")
                    history_output = gr.Dataframe(
                        label="Prediction History",
                        interactive=False
                    )
            
            gr.Markdown("""
            ---
            ### ✨ System Features
            
            | Feature | Description |
            |---------|-------------|
            | 🚀 **Fast Performance** | QLoRA with 4-bit quantization + intelligent caching |
            | 🎨 **Modern UI** | Clean, intuitive interface with real-time feedback |
            | 📊 **Batch Processing** | Process multiple products at once via CSV |
            | 📈 **Analytics** | Visualize pricing trends and patterns |
            | 💾 **Smart Caching** | Instant results for repeated queries |
            | 🕒 **History Tracking** | Monitor and analyze all predictions |
            
            **Model:** TinyLlama-1.1B with QLoRA fine-tuning  
            **Quantization:** 4-bit NF4 for memory efficiency  
            **Inference Time:** <1 second per prediction  
            """)
            
            # Event handlers
            predict_btn.click(
                fn=self.predict_price,
                inputs=[category, brand, condition, age_months, 
                       rating, num_reviews, weight_kg],
                outputs=[price_output, response_output, cache_status]
            )
            
            clear_btn.click(
                fn=lambda: ("", "", ""),
                outputs=[price_output, response_output, cache_status]
            )
            
            clear_cache_btn.click(
                fn=self.clear_cache,
                outputs=cache_msg
            )
            
            batch_btn.click(
                fn=self.process_batch,
                inputs=batch_file,
                outputs=batch_output
            )
            
            analytics_btn.click(
                fn=self.generate_analytics,
                outputs=analytics_plot
            )
            
            history_btn.click(
                fn=self.get_history_df,
                outputs=history_output
            )
            
            clear_hist_btn.click(
                fn=self.clear_history,
                outputs=history_output
            )
        
        return demo
