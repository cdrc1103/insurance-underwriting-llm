"""Model-specific configuration constants."""

# Model identification
DEFAULT_MODEL_NAME: str = "Qwen/Qwen3-0.6B"

# Context length limits
QWEN_MAX_gCONTEXT: int = 32768  # Qwen3 native context limit (32K tokens)

# Fine-tuning recommendations for T4 16GB GPU
# Updated based on token analysis (User Story 1.5) - 95th percentile of training set
MAX_TOKEN_LENGTH: int = 21486  # Covers 95% of examples without truncation

# Generation parameters
DEFAULT_MAX_NEW_TOKENS: int = 512
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_P: float = 0.9
DEFAULT_TOP_K: int = 50
DEFAULT_REPETITION_PENALTY: float = 1.1
