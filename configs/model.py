"""Model-specific configuration constants."""

# Model identification
DEFAULT_MODEL_NAME: str = "Qwen/Qwen3-0.6B"

# Context length limits
QWEN_MAX_CONTEXT: int = 32768  # Qwen3 native context limit (32K tokens)

# Fine-tuning recommendations for T4 16GB GPU
# Updated based on token analysis (User Story 1.5) - 95th percentile of training set
RECOMMENDED_FINE_TUNING_MAX_LENGTH: int = 21486  # Covers 95% of examples without truncation
