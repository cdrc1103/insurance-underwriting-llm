# Qwen3 Fine-tuning Data Format

## Overview

This document summarizes the expected data format for fine-tuning Qwen3-0.6B on the insurance underwriting task.

## ChatML Template

Qwen3 uses **ChatML format** with special control tokens:

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_response}<|im_end|>
```

### Special Tokens
- `<|im_start|>` - Start of a message turn
- `<|im_end|>` - End of a message turn
- Roles: `system`, `user`, `assistant`

### Important Notes
- The `<|im_start|>` and `<|im_end|>` tokens are **trained in the Instruct model** but untrained in base models
- Do NOT set `pad_token` to `<|endoftext|>` - this causes infinite generation issues
- Do NOT set `bos_token` to `<|im_start|>` - causes duplication

## Qwen3 Thinking Mode

Qwen3-0.6B supports **hybrid thinking mode** via the `<think>...</think>` tags:

### Enabling Thinking Mode

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # Default; set to False to disable
)
```

### Thinking Mode Format

When thinking mode is enabled, the assistant response includes internal reasoning:

```
<|im_start|>assistant
<think>Let me analyze the company profile and check guidelines...</think>
Based on my analysis, I recommend the following products...<|im_end|>
```

### Soft Switch via User Input

Add `/think` or `/no_think` to prompts for dynamic control:

```python
user_input = "What coverage do they need? /think"
```

## Dataset Format Options

### Option 1: Messages Format (Recommended for SFTTrainer)

```json
{
  "messages": [
    {"role": "system", "content": "You are an insurance underwriting assistant..."},
    {"role": "user", "content": "What insurance products suit this company?"},
    {"role": "assistant", "content": "<think>internal reasoning</think>Based on the company profile..."}
  ]
}
```

### Option 2: ShareGPT Format (LLaMA-Factory compatible)

```json
{
  "conversations": [
    {"from": "human", "value": "user instruction"},
    {"from": "gpt", "value": "model response"}
  ],
  "system": "system prompt (optional)",
  "tools": "tool description (optional)"
}
```

### Option 3: Alpaca Format

```json
{
  "instruction": "user instruction",
  "input": "additional context (optional)",
  "output": "model response",
  "system": "system prompt (optional)",
  "history": [
    ["user turn 1", "assistant turn 1"],
    ["user turn 2", "assistant turn 2"]
  ]
}
```

## Tool Calling Format

Qwen3 supports function calling with this format in assistant responses:

```xml
<tool_call>
{"name": "function_name", "arguments": {"arg1": "value1"}}
</tool_call>
```

Tool results are provided as messages with role `tool` or `function`:
```json
{"role": "tool", "content": "tool result here"}
```

## Recommended Format for This Project

Given our insurance underwriting use case with tool calls and internal reasoning, the recommended format is:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an insurance underwriting assistant. You help underwriters assess company eligibility, recommend insurance products, and make coverage decisions.\n\nCompany Profile:\n- Name: {company_name}\n- Annual Revenue: ${revenue}\n- Employees: {employees}\n- State: {state}\n- Description: {description}\n\nYou have access to the following tool:\n- get_underwriting_guidelines: Retrieve underwriting guidelines and appetite information"
    },
    {
      "role": "user",
      "content": "What insurance products would be suitable for this company?"
    },
    {
      "role": "assistant",
      "content": "<think>Let me check the underwriting guidelines for this company type.</think><tool_call>\n{\"name\": \"get_underwriting_guidelines\", \"arguments\": {}}\n</tool_call>"
    },
    {
      "role": "tool",
      "content": "***General***\nWe write policies for small businesses only..."
    },
    {
      "role": "assistant",
      "content": "<think>The guidelines indicate this is a small business that qualifies for coverage.</think>Based on the underwriting guidelines, I recommend the following products..."
    }
  ]
}
```

## Training Considerations

### Loss Masking
When using SFTTrainer, use `DataCollatorForCompletionOnlyLM` to only train on assistant responses:

```python
from trl import DataCollatorForCompletionOnlyLM

collator = DataCollatorForCompletionOnlyLM(
    response_template='<|im_start|>assistant\n',
    tokenizer=tokenizer
)
```

### Sequence Length
- Qwen3-0.6B supports up to 32K context
- For fine-tuning on T4 16GB: recommend max_seq_length of 2048-4096
- Our dataset: median ~1400 tokens, max ~40K tokens (some examples may need truncation)

### apply_chat_template
The tokenizer can automatically format messages:

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # Enable thinking mode for reasoning
)
```

### Thinking Mode Considerations
- Use `enable_thinking=True` to train the model to produce internal reasoning
- The `<think>...</think>` tags help separate internal reasoning from user-facing responses
- Training with thinking mode helps maintain chain-of-thought reasoning capabilities

## Sources

- [Qwen3-0.6B Model Card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)
- [Qwen Key Concepts](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)
- [Qwen Function Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [LLaMA-Factory Training Guide](https://qwen.readthedocs.io/en/v2.0/training/SFT/llama_factory.html)
- [MS-SWIFT Training Guide](https://qwen.readthedocs.io/en/latest/training/ms_swift.html)
- [Unsloth Qwen Fine-tuning Docs](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune)
