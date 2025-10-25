import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    """Configuration for model training and inference"""
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Fast, lightweight model
    max_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    output_dir: str = "./models/price_predictor"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    max_grad_norm: float = 0.3
    weight_decay: float = 0.001

@dataclass
class CacheConfig:
    """Caching configuration"""
    use_cache: bool = True
    cache_backend: str = "memory"  # "memory" for simplicity
    cache_ttl: int = 3600  # 1 hour
