import os
import sys
from config import ModelConfig, TrainingConfig, CacheConfig
from data_generator import PriceDataGenerator
from model_trainer import PricePredictorTrainer
from inference_engine import PricePredictor
from app import PricePredictionApp

def setup_directories():
    """Create necessary directories"""
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    print("✓ Directories created")

def train_model():
    """Train the price prediction model"""
    print("\n" + "="*70)
    print("🚀 STEP 1: DATA GENERATION")
    print("="*70)
    
    # Generate dataset
    generator = PriceDataGenerator(num_samples=5000)
    df = generator.generate_dataset()
    instruction_data = generator.create_instruction_dataset(df)
    
    # Save dataset
    df.to_csv("./data/product_dataset.csv", index=False)
    print(f"✓ Generated {len(instruction_data)} training samples")
    print(f"✓ Dataset saved to ./data/product_dataset.csv")
    
    # Split data (90% train, 10% eval)
    train_size = int(0.9 * len(instruction_data))
    train_data = instruction_data[:train_size]
    eval_data = instruction_data[train_size:]
    
    print(f"✓ Train samples: {len(train_data)}")
    print(f"✓ Eval samples: {len(eval_data)}")
    
    print("\n" + "="*70)
    print("🔧 STEP 2: MODEL FINE-TUNING WITH QLORA")
    print("="*70)
    
    # Initialize configs
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    print(f"Base Model: {model_config.base_model}")
    print(f"LoRA Rank (r): {model_config.lora_r}")
    print(f"LoRA Alpha: {model_config.lora_alpha}")
    print(f"Training Epochs: {training_config.num_epochs}")
    print(f"Batch Size: {training_config.batch_size}")
    
    # Initialize trainer
    trainer = PricePredictorTrainer(model_config, training_config)
    
    # Load model and tokenizer
    trainer.load_model_and_tokenizer()
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = trainer.prepare_dataset(train_data)
    eval_dataset = trainer.prepare_dataset(eval_data)
    print("✓ Datasets prepared")
    
    # Train
    trainer.train(train_dataset, eval_dataset)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)
    print(f"Model saved to: {training_config.output_dir}")

def launch_app():
    """Launch the Gradio UI"""
    print("\n" + "="*70)
    print("🎨 STEP 3: LAUNCHING UI")
    print("="*70)
    
    training_config = TrainingConfig()
    cache_config = CacheConfig()
    
    # Check if model exists
    if not os.path.exists(training_config.output_dir):
        print(f"❌ Error: Model not found at {training_config.output_dir}")
        print("Please run training first: python main.py --train")
        sys.exit(1)
    
    # Load predictor
    print("Loading model for inference...")
    predictor = PricePredictor(training_config.output_dir, cache_config)
    
    # Create app
    print("Creating Gradio interface...")
    app = PricePredictionApp(predictor)
    demo = app.create_ui()
    
    # Launch
    print("\n✓ Launching application...")
    print("="*70)
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

def main():
    """Main execution flow"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Price Prediction System")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["train", "app", "all"],
        help="Execution mode: train (fine-tune model), app (launch UI), all (both)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎯 AI PRICE PREDICTION SYSTEM")
    print("Fine-tuned LLM with QLoRA")
    print("="*70)
    
    setup_directories()
    
    if args.mode in ["train", "all"]:
        train_model()
    
    if args.mode in ["app", "all"]:
        launch_app()

if __name__ == "__main__":
    main()