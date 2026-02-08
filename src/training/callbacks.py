"""Custom training callbacks for logging and monitoring."""

import logging

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src.utils.memory import cleanup_cuda_memory, get_gpu_memory_stats

logger = logging.getLogger(__name__)


class MemoryLoggingCallback(TrainerCallback):
    """
    Logs GPU memory usage and performs cleanup during training.

    Logs memory statistics at each logging step and cleans up CUDA
    memory after each evaluation to prevent gradual accumulation.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Log GPU memory at each logging step."""
        if not torch.cuda.is_available():
            return

        stats = get_gpu_memory_stats()
        logger.info(
            f"[Step {state.global_step}] GPU Memory: "
            f"{stats['allocated_mb']:.0f}MB allocated, "
            f"{stats['reserved_mb']:.0f}MB reserved"
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Clean up GPU memory after evaluation."""
        cleanup_cuda_memory()


class SampleGenerationCallback(TrainerCallback):
    """
    Generates sample responses at the end of each epoch for qualitative monitoring.

    Args:
        tokenizer: Tokenizer for decoding
        sample_prompts: List of message lists to generate responses for
        max_new_tokens: Maximum tokens to generate per sample
        num_samples: Number of samples to generate per epoch
    """

    def __init__(
        self,
        tokenizer,
        sample_prompts: list[list[dict[str, str]]],
        max_new_tokens: int = 128,
        num_samples: int = 3,
    ):
        self.tokenizer = tokenizer
        self.sample_prompts = sample_prompts[:num_samples]
        self.max_new_tokens = max_new_tokens

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        """Generate sample responses at epoch end."""
        if model is None:
            return

        model.eval()
        epoch = int(state.epoch) if state.epoch is not None else 0
        logger.info(f"\n--- Sample generations (epoch {epoch}) ---")

        for i, messages in enumerate(self.sample_prompts):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(prompt, return_tensors="pt")
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                # Decode only new tokens
                new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                logger.info(f"Sample {i + 1}: {response[:200]}...")

            except (RuntimeError, ValueError, IndexError) as e:
                logger.warning(f"Sample generation {i + 1} failed: {e}")

        model.train()
