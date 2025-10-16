import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from typing import Dict, List
from config import ModelConfig, TrainingConfig

class PricePredictorTrainer:
    """Fine-tune DistilGPT-2 with LoRA for price prediction"""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def load_model_and_tokenizer(self):
        """Load DistilGPT-2 model and tokenizer with LoRA configuration"""
        # ✅ Use smaller base model
        self.model_config.base_model = "distilgpt2"

        print(f"Loading tokenizer from {self.model_config.base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        print(f"Loading DistilGPT-2 model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        model = model.to(self.device)
        model.config.use_cache = False

        # ✅ LoRA configuration for DistilGPT-2
        lora_config = LoraConfig(
            r=self.model_config.lora_r,
            lora_alpha=self.model_config.lora_alpha,
            target_modules=self.model_config.target_modules,
            lora_dropout=self.model_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
            init_lora_weights=True
        )

        print("Adding LoRA adapters...")
        self.model = get_peft_model(model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(self, data: List[Dict]) -> Dataset:
        """Prepare dataset for training"""
        def format_instruction(sample):
            prompt = f"""<|system|>
You are a price prediction expert. Analyze product details and provide accurate price estimates.
<|user|>
{sample['instruction']}
<|assistant|>
{sample['output']}"""
            return prompt

        formatted_data = [{"text": format_instruction(item)} for item in data]
        dataset = Dataset.from_list(formatted_data)

        def tokenize(sample):
            result = self.tokenizer(
                sample["text"],
                truncation=True,
                max_length=self.model_config.max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized = dataset.map(tokenize, remove_columns=["text"])
        return tokenized

    def train(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """Train the DistilGPT-2 model with LoRA"""
        import os
        os.makedirs(self.training_config.output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            warmup_steps=self.training_config.warmup_steps,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            max_grad_norm=self.training_config.max_grad_norm,
            weight_decay=self.training_config.weight_decay,
            fp16=False,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            report_to="none",
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        print("\n🚀 Starting training...")
        trainer.train()

        print(f"\n💾 Saving model to {self.training_config.output_dir}...")
        self.model.save_pretrained(self.training_config.output_dir)
        self.tokenizer.save_pretrained(self.training_config.output_dir)

        return trainer
