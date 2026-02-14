"""Model-specific configuration constants."""

# Model identification
DEFAULT_MODEL_NAME: str = "Qwen/Qwen3-0.6B"

# Context length limits
QWEN_MAX_CONTEXT: int = 32768  # Qwen3 native context limit (32K tokens)

# Fine-tuning recommendations for T4 16GB GPU
# Updated based on token analysis (User Story 1.5) - 95th percentile of training set
MAX_TOKEN_LENGTH: int = 21486  # Covers 95% of examples without truncation

# Generation parameters
DEFAULT_MAX_NEW_TOKENS: int = 512
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_P: float = 0.9
DEFAULT_TOP_K: int = 50
DEFAULT_REPETITION_PENALTY: float = 1.1
DEFAULT_USE_CACHE: bool = True  # Disable KV cache if you encounter OOM errors on GPU

# Training hyperparameters (from Phase 1 analysis - results/hyperparameters.json)
DEFAULT_LEARNING_RATE: float = 2e-4
DEFAULT_TRAIN_BATCH_SIZE: int = 4
DEFAULT_GRADIENT_ACCUMULATION_STEPS: int = 2
DEFAULT_NUM_EPOCHS: int = 3
DEFAULT_WARMUP_STEPS: int = 8
DEFAULT_WEIGHT_DECAY: float = 0.01
DEFAULT_MAX_GRAD_NORM: float = 1.0
DEFAULT_SCHEDULER: str = "cosine"
DEFAULT_SEED: int = 42

# LoRA configuration
DEFAULT_LORA_R: int = 16
DEFAULT_LORA_ALPHA: int = 32
DEFAULT_LORA_DROPOUT: float = 0.05
DEFAULT_LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Early stopping
DEFAULT_EARLY_STOPPING_PATIENCE: int = 3

# Logging and checkpointing intervals (steps_per_epoch = 27)
DEFAULT_LOGGING_STEPS: int = 5
DEFAULT_SAVE_STEPS: int = 27
DEFAULT_EVAL_STEPS: int = 27
