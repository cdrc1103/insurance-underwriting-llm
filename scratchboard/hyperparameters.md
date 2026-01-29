# Hyperparameter Calculations: Theoretical Background

This document explains the theoretical reasoning behind each hyperparameter calculation in [scripts/calculate_hyperparameters.py](../scripts/calculate_hyperparameters.py).

## Table of Contents
- [Memory Calculations](#memory-calculations)
- [Batch Size Optimization](#batch-size-optimization)
- [Training Schedule](#training-schedule)
- [Learning Rate Scaling](#learning-rate-scaling)
- [Optimizer Configuration (AdamW)](#optimizer-configuration-adamw)
- [Learning Rate Scheduler](#learning-rate-scheduler)
- [LoRA Configuration](#lora-configuration)
- [Mixed Precision Training](#mixed-precision-training)

---

## Memory Calculations

### 1. Model Memory (FP16)

**Formula:**
```python
model_memory_gb = (model_params_millions * 2) / 1024
```

**Theoretical Background:**

When storing model parameters in FP16 (16-bit floating point) precision:
- Each parameter requires **2 bytes** (16 bits ÷ 8 bits/byte)
- For a model with M million parameters: `M × 10^6 × 2 bytes`
- Convert to GB: divide by 1024 MB/GB

**Why FP16?**
- Reduces memory footprint by 50% compared to FP32
- Modern GPUs (NVIDIA Volta+, AMD RDNA2+) have hardware acceleration for FP16 operations
- Minimal accuracy loss for most deep learning tasks
- Enables training larger models or larger batch sizes

**References:**
- Mixed precision training: Micikevicius et al. (2017), "Mixed Precision Training"

---

### 2. LoRA Parameters

**Formula:**
```python
lora_params_millions = (2 * lora_r * d_model * num_layers * num_target_modules) / 1e6
```

**Theoretical Background:**

LoRA (Low-Rank Adaptation) decomposes weight updates as: `ΔW = BA`, where:
- `B`: matrix of size `d_model × r`
- `A`: matrix of size `r × d_model`
- `r`: rank (bottleneck dimension)

For each target module (q_proj, k_proj, etc.):
- Parameters added = `d_model × r + r × d_model = 2 × d_model × r`
- Multiply by number of layers and target modules

**Why LoRA?**
- Dramatically reduces trainable parameters (often 1000x reduction)
- Original insight: most weight updates lie in a low-dimensional subspace
- Enables fine-tuning large models on consumer GPUs
- No inference latency penalty (can merge weights back)

**Example (Qwen3-0.6B):**
- d_model = 896, r = 16, 24 layers, 7 target modules
- LoRA params = 2 × 16 × 896 × 24 × 7 ≈ 4.8M parameters
- Original model = 619M parameters → only 0.77% trainable!

**References:**
- Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models"

---

### 3. Optimizer State Memory (AdamW)

**Formula:**
```python
optimizer_memory_gb = (lora_params_millions * 8) / 1024
```

**Theoretical Background:**

AdamW optimizer maintains per-parameter state:
- **First moment** (momentum): exponential moving average of gradients
- **Second moment** (variance): exponential moving average of squared gradients

Storage requirements:
- Each state: 4 bytes (FP32) per trainable parameter
- Total: 2 states × 4 bytes = **8 bytes per parameter**

**Why FP32 for optimizer states?**
- Optimizer states accumulate small updates over many steps
- FP16 lacks precision for these tiny accumulated values
- Mixed precision keeps states in FP32 for numerical stability

**Memory breakdown example:**
- 4.8M LoRA parameters × 8 bytes = 38.4 MB for optimizer states
- Compare to full model: 619M × 8 = 4.95 GB (130x more!)

**References:**
- Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization"

---

### 4. Activation Memory Per Example

**Formula:**
```python
activation_memory_bytes = seq_length * num_layers * 12
```

**Theoretical Background:**

Activations are intermediate values stored during forward pass for backward pass gradient computation.

**Per layer, per token, we store:**
- Query, Key, Value projections (3 × hidden_dim × 4 bytes)
- Attention scores (seq_length × 4 bytes)
- MLP intermediate activations (ffn_dim × 4 bytes)
- Residual connections and layer norms

**Approximation: 12 bytes per token per layer**
- Conservative estimate covering all intermediate activations
- Scales linearly with sequence length and number of layers
- Dominant memory consumer during training

**Why activations dominate:**
- For batch_size=1, seq_length=512, 24 layers:
  - Activations: 512 × 24 × 12 = 147 KB per example
  - With batch_size=4: 147 × 4 = 588 KB
- Activations grow with batch size, model weights don't

**Memory optimization strategies:**
- Gradient checkpointing: recompute activations instead of storing (trades compute for memory)
- Flash Attention: reduces attention memory from O(N²) to O(N)

**References:**
- Chen et al. (2016), "Training Deep Nets with Sublinear Memory Cost"

---

## Batch Size Optimization

### 5. Safety Margin

**Formula:**
```python
available_memory = (gpu_memory_gb - model_memory_gb) * safety_margin
safety_margin = 0.85  # 85% utilization
```

**Theoretical Background:**

**Why not use 100% GPU memory?**

1. **Memory fragmentation**: GPU memory allocators can't perfectly pack allocations
2. **Framework overhead**: PyTorch/CUDA runtime needs working space
3. **Dynamic allocations**: Temporary buffers during backward pass
4. **Safety buffer**: Prevents OOM crashes mid-training

**Why 85%?**
- Industry standard for production training
- Balances utilization vs. stability
- Leaves ~15% headroom for memory spikes

**Consequences of OOM:**
- Training crashes, losing progress
- GPU state requires reset
- Wasted compute time

**References:**
- Best practices from NVIDIA, PyTorch documentation

---

### 6. Optimal Batch Size

**Formula:**
```python
batch_size = max(1, int(available_memory / memory_per_example))
batch_size = min(batch_size, 4)  # Cap at 4
```

**Theoretical Background:**

**Calculation:**
- Divide available memory by per-example cost
- Ensures activations fit in memory
- Floor to integer (can't have fractional examples)

**Why cap at 4?**

1. **Training stability**:
   - Very large batches can hurt generalization
   - Small batches add beneficial noise to gradients

2. **Hardware efficiency**:
   - T4 GPU optimized for batch sizes 1-4
   - Larger batches don't proportionally increase throughput

3. **Gradient accumulation alternative**:
   - Better to use batch_size=4 + accumulation than batch_size=32
   - More frequent gradient updates improves convergence

**Trade-offs:**
- Larger batches: more stable gradients, better GPU utilization
- Smaller batches: more gradient updates, better generalization, more noise

**References:**
- Masters & Luschi (2018), "Revisiting Small Batch Training for Deep Neural Networks"
- Smith et al. (2018), "Don't Decay the Learning Rate, Increase the Batch Size"

---

### 7. Gradient Accumulation

**Formula:**
```python
target_effective_batch = 8
grad_accum_steps = max(1, target_effective_batch // batch_size)
effective_batch_size = batch_size * grad_accum_steps
```

**Theoretical Background:**

**Problem:** GPU memory limits physical batch size, but optimal training needs larger effective batches.

**Solution:** Accumulate gradients over multiple mini-batches before updating weights.

**Algorithm:**
```
1. Zero gradients
2. For i in range(grad_accum_steps):
   - Forward pass on mini-batch
   - Backward pass (gradients accumulate)
3. Average accumulated gradients
4. Optimizer step (update weights)
```

**Mathematical equivalence:**
- Gradient accumulation over N steps ≈ batch size of N × mini_batch
- G_total = (G₁ + G₂ + ... + Gₙ) / N
- Same weight update as processing all examples together

**Why target_effective_batch = 8?**
- Empirically good for fine-tuning small LLMs (0.5-3B parameters)
- Balances gradient stability with training speed
- Allows meaningful gradient signal from diverse examples

**Trade-offs:**
- Memory: constant (only stores 1 mini-batch)
- Speed: slightly slower than true large batch (overhead from multiple passes)
- Convergence: mathematically equivalent to large batch

**References:**
- Ott et al. (2018), "Scaling Neural Machine Translation"

---

## Training Schedule

### 8. Number of Epochs

**Value:** `num_epochs = 3`

**Theoretical Background:**

**Why 3 epochs for fine-tuning?**

1. **Transfer learning principle**: Pre-trained model already knows language
   - Just adapting to task-specific patterns
   - Don't need many epochs to learn domain knowledge

2. **Overfitting risk**:
   - Small fine-tuning datasets (thousands, not millions of examples)
   - More epochs → memorization of training data
   - Hurts generalization to test data

3. **Catastrophic forgetting**:
   - Too many epochs can "overwrite" pre-trained knowledge
   - Model forgets general language understanding
   - Becomes too specialized

4. **Empirical results**:
   - Research shows 2-4 epochs optimal for most fine-tuning tasks
   - Diminishing returns after 3 epochs
   - Validation loss often increases after epoch 3

**Contrast with pre-training:**
- Pre-training: 1-3 epochs on massive datasets (billions of tokens)
- Fine-tuning: 3-5 epochs on small datasets (thousands of examples)

**References:**
- Devlin et al. (2019), BERT fine-tuning guidelines
- HuggingFace Transformers documentation

---

### 9. Warmup Steps

**Formula:**
```python
warmup_steps = int(total_steps * 0.1)
warmup_ratio = 0.1  # 10%
```

**Theoretical Background:**

**What is warmup?**
- Gradually increase learning rate from 0 → target LR over first N steps
- Prevents early training instability

**Why is warmup needed?**

1. **Random initialization of new layers**:
   - LoRA matrices initialized near zero
   - Large gradients at start can destabilize training
   - Warm start allows careful adjustment

2. **Optimizer state initialization**:
   - Adam's moving averages start at zero
   - Need time to accumulate meaningful statistics
   - Large LR + uninitialized stats = unstable updates

3. **Loss landscape exploration**:
   - Early gradients can be noisy and large
   - Warm start prevents jumping to bad local minima
   - Allows smooth descent into good basin

**Why 10% warmup ratio?**
- Empirical sweet spot for transformer fine-tuning
- Too short (<5%): instability remains
- Too long (>20%): wastes training on suboptimal LR
- 10% gives stable start without sacrificing training time

**Warmup schedule (linear):**
```
LR(step) = target_LR * (step / warmup_steps)  for step ≤ warmup_steps
```

**References:**
- Vaswani et al. (2017), "Attention is All You Need" (original warmup proposal)
- Goyal et al. (2017), "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"

---

## Learning Rate Scaling

### 10. Base Learning Rate

**Value:** `base_lr = 2e-4`

**Theoretical Background:**

**Why 2e-4 (0.0002) for LoRA fine-tuning?**

1. **LoRA-specific tuning**:
   - LoRA adapters are randomly initialized
   - Need larger LR than pre-trained weights would tolerate
   - But smaller than training from scratch (1e-3 to 1e-2)

2. **Empirical validation**:
   - Hu et al. (2021) recommend 1e-4 to 3e-4 for LoRA
   - 2e-4 is middle ground, works across diverse tasks
   - Rarely needs task-specific tuning

3. **Comparison to other methods**:
   - Full fine-tuning: 1e-5 to 5e-5 (much smaller, adjust pre-trained weights)
   - LoRA: 1e-4 to 3e-4 (larger, train new adapters)
   - Training from scratch: 1e-3 to 1e-2 (largest, all weights random)

**Why not higher?**
- Risk of catastrophic forgetting
- Can destabilize training (loss spikes)
- Might overshoot optimal solution

**Why not lower?**
- Slow convergence
- May not fully adapt to new task
- Wastes compute time

**References:**
- Hu et al. (2021), "LoRA" paper, Appendix B

---

### 11. Learning Rate Scaling with Batch Size

**Formula:**
```python
learning_rate = base_lr * (effective_batch_size / 8) ** 0.5
```

**Theoretical Background:**

**The batch size-LR relationship:**

When increasing batch size, you must adjust learning rate to maintain equivalent optimization behavior.

**Linear scaling rule** (Goyal et al. 2017):
```
LR_new = LR_base * (batch_new / batch_base)
```
- Used for very large-scale training (batch size 256-8192)
- Assumes gradients are roughly independent

**Square root scaling rule** (Hoffer et al. 2017):
```
LR_new = LR_base * sqrt(batch_new / batch_base)
```
- More conservative
- Better for smaller batch sizes (8-64)
- Accounts for gradient correlation within batch

**Why sqrt scaling for our use case?**

1. **Small batch regime**: effective_batch_size = 8-16
2. **Correlated gradients**: insurance domain text has patterns
3. **Fine-tuning stability**: over-scaling LR risks destabilization

**Example:**
```
base_lr = 2e-4, reference_batch = 8

batch=4:  LR = 2e-4 * sqrt(4/8)  = 1.41e-4  (30% reduction)
batch=8:  LR = 2e-4 * sqrt(8/8)  = 2.00e-4  (no change)
batch=16: LR = 2e-4 * sqrt(16/8) = 2.83e-4  (41% increase)
```

**Intuition:**
- Larger batches → more stable gradient estimates → can take larger steps
- Smaller batches → noisier gradients → need smaller steps for stability

**References:**
- Goyal et al. (2017), "Accurate, Large Minibatch SGD"
- Hoffer et al. (2017), "Train longer, generalize better"
- You et al. (2017), "Scaling SGD Batch Size to 32K"

---

## Optimizer Configuration (AdamW)

### 12. AdamW vs. Adam

**Choice:** `optimizer = "AdamW"`

**Theoretical Background:**

**Adam (Adaptive Moment Estimation):**
- Combines momentum (first moment) + adaptive learning rates (second moment)
- Update rule:
  ```
  m_t = β₁ * m_{t-1} + (1-β₁) * g_t          # momentum
  v_t = β₂ * v_{t-1} + (1-β₂) * g_t²         # variance
  θ_t = θ_{t-1} - lr * m_t / (sqrt(v_t) + ε)  # update
  ```

**AdamW (Adam with Weight Decay):**
- Fixes Adam's weight decay implementation
- Original Adam applied L2 penalty to gradients (incorrect)
- AdamW applies weight decay directly to weights:
  ```
  θ_t = θ_{t-1} * (1 - λ) - lr * m_t / (sqrt(v_t) + ε)
  ```

**Why AdamW for transformers?**
1. Better regularization: decouples weight decay from gradient-based updates
2. Empirically better generalization on NLP tasks
3. Standard optimizer for all modern transformer training
4. More stable across different learning rates

**References:**
- Kingma & Ba (2014), "Adam: A Method for Stochastic Optimization"
- Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization"

---

### 13. Adam Beta Parameters

**Values:**
```python
adam_beta1 = 0.9
adam_beta2 = 0.999
```

**Theoretical Background:**

**Beta₁ (momentum decay rate):**
- Controls exponential moving average of gradients (first moment)
- β₁ = 0.9 → 90% of previous momentum + 10% current gradient
- Effective window: ~10 steps (1/(1-β₁))

**Why β₁ = 0.9?**
- Standard value from original Adam paper
- Provides smoothing over ~10 gradient updates
- Helps overcome local minima and saddle points
- Too high (>0.95): slow to respond to changes
- Too low (<0.8): too noisy, doesn't build momentum

**Beta₂ (variance decay rate):**
- Controls exponential moving average of squared gradients (second moment)
- β₂ = 0.999 → 99.9% of previous variance + 0.1% current squared gradient
- Effective window: ~1000 steps (1/(1-β₂))

**Why β₂ = 0.999?**
- Long-term memory of gradient magnitudes
- Provides stable adaptive learning rates
- Essential for transformers (very different scales across layers)
- Lower values (0.99) can work but less stable

**Relationship:**
- β₂ >> β₁ is critical
- Longer memory for variance (stability) than momentum (responsiveness)

**Special cases:**
- Sparse gradients: use β₂ = 0.999 (default)
- Dense gradients: can use β₂ = 0.99 for faster adaptation

**References:**
- Kingma & Ba (2014), extensive experiments on β values
- Transformer literature consistently uses (0.9, 0.999)

---

### 14. Adam Epsilon

**Value:** `adam_epsilon = 1e-8`

**Theoretical Background:**

**Purpose:**
- Numerical stability term in denominator
- Prevents division by zero when variance estimate is tiny

**Update rule:**
```
θ_t = θ_{t-1} - lr * m_t / (sqrt(v_t) + ε)
                              ^^^^^^^^^^
```

**Why 1e-8?**
1. **Small enough**: doesn't affect typical variance values (usually > 1e-4)
2. **Large enough**: prevents true zero division
3. **Float32 precision**: ~7 decimal digits, 1e-8 is safely above underflow
4. **Standard value**: from original Adam paper

**When epsilon matters:**
- Rarely updated parameters (sparse features)
- Very small gradients (well-trained regions)
- Prevents NaN propagation

**Too large (1e-4):** acts as additional learning rate reduction
**Too small (1e-12):** risk of numerical instability in FP16

**References:**
- Kingma & Ba (2014), default value used in all experiments

---

### 15. Weight Decay

**Value:** `weight_decay = 0.01`

**Theoretical Background:**

**What is weight decay?**
- L2 regularization applied directly to weights
- Shrinks weights toward zero: `θ_t = θ_{t-1} * (1 - λ)`
- λ = weight_decay coefficient

**Purpose:**
1. **Prevents overfitting**: penalizes large weights
2. **Improves generalization**: smoother decision boundaries
3. **Implicit feature selection**: unimportant weights decay to zero

**Why 0.01 for LoRA?**
- Standard value from transformer literature
- Effective for regularizing adapters without over-constraining
- Balance between regularization and learning capacity

**Comparison:**
- Pre-training: 0.01-0.1 (strong regularization)
- Fine-tuning: 0.01 (moderate regularization)
- LoRA: 0.01 (match fine-tuning; adapters are small already)

**What gets decayed?**
- All weights EXCEPT: biases, layer norm parameters
- Rationale: bias terms don't increase model capacity

**Mathematical effect:**
```
Effective learning rate = base_LR * (1 - weight_decay)
For weight_decay=0.01: weights retain 99% per step
```

**References:**
- Loshchilov & Hutter (2019), AdamW paper recommends 0.01
- Transformers library defaults

---

### 16. Gradient Clipping

**Value:** `max_grad_norm = 1.0`

**Theoretical Background:**

**What is gradient clipping?**
- Caps the L2 norm of the gradient vector
- If ||g|| > max_norm: scale g → g * (max_norm / ||g||)
- Prevents exploding gradients

**Algorithm:**
```python
total_norm = sqrt(sum(g_i^2 for all gradients g_i))
if total_norm > max_grad_norm:
    scale = max_grad_norm / total_norm
    for g in gradients:
        g *= scale
```

**Why is this needed?**

1. **Transformer instability**:
   - Attention mechanism can produce very large gradients
   - Deep networks amplify gradient magnitude
   - Single bad batch can derail training

2. **Mixed precision training**:
   - FP16 has limited range: [6e-5, 65504]
   - Gradient overflow → NaN → training collapse
   - Clipping prevents overflow

3. **LoRA considerations**:
   - Adapter initialization creates large early gradients
   - Clipping stabilizes first few hundred steps

**Why max_grad_norm = 1.0?**
- Standard value from transformer pre-training (BERT, GPT)
- Empirically optimal for Adam optimizer
- Strict enough to prevent instability
- Loose enough to not impede learning

**Comparison:**
- Pre-training: 1.0 (strict control)
- Fine-tuning: 1.0 (maintain stability)
- Vision models: 5.0-10.0 (more tolerance)

**Side effects:**
- Effectively reduces learning rate on large-gradient batches
- Can slow convergence if set too low (<0.5)
- Essential for FP16 training

**References:**
- Pascanu et al. (2013), "On the difficulty of training RNNs"
- BERT, GPT-2, GPT-3 all use 1.0

---

## Learning Rate Scheduler

### 17. Cosine Annealing Schedule

**Configuration:**
```python
scheduler = "cosine"
min_lr_ratio = 0.1
```

**Theoretical Background:**

**Schedule formula:**
```python
LR(t) = LR_min + (LR_max - LR_min) * 0.5 * (1 + cos(π * t / T))
```
where:
- t = current step (after warmup)
- T = total steps
- LR_max = peak learning rate
- LR_min = min_lr_ratio * LR_max = 0.1 * LR_max

**Shape:**
- Smooth decay from LR_max → LR_min
- Fast descent early (coarse search)
- Slow descent late (fine-tuning)
- Never reaches exactly zero (maintains some learning)

**Why cosine schedule?**

1. **Smooth convergence**:
   - Gradual decay prevents abrupt changes
   - Better than step decay (sudden drops can destabilize)

2. **Exploration-exploitation trade-off**:
   - High LR (early): explore loss landscape
   - Medium LR (mid): converge to basin
   - Low LR (late): fine-tune within basin

3. **Empirically superior**:
   - Consistently outperforms step decay in transformers
   - Standard in modern NLP (BERT, GPT, T5, etc.)

4. **No hyperparameter tuning**:
   - Step decay requires choosing decay milestones
   - Cosine is parameter-free (except min_lr_ratio)

**Why min_lr_ratio = 0.1?**
- Maintains 10% of peak LR at end of training
- Prevents complete stagnation (0 LR = no learning)
- Allows continued adaptation in final epochs
- Too low (<0.01): training effectively stops
- Too high (>0.3): insufficient annealing

**Alternative schedules:**
- **Linear decay**: simpler but sharp transitions at boundaries
- **Step decay**: requires manual milestone selection
- **Polynomial decay**: similar to cosine but less smooth
- **Constant LR**: no decay, risks overfitting

**Cosine with warmup combined:**
```
       warmup              cosine decay
    /|         |------------------------\
   / |         |                         \
  /  |         |                          \___
 /   |         |                              \
0   10%       100%                          end
    steps
```

**References:**
- Loshchilov & Hutter (2016), "SGDR: Stochastic Gradient Descent with Warm Restarts"
- Used in BERT, GPT-2, T5, and virtually all modern transformers

---

## LoRA Configuration

### 18. LoRA Rank (r)

**Value:** `lora_r = 16`

**Theoretical Background:**

**What is rank?**
- Dimension of the bottleneck in the low-rank decomposition
- ΔW = BA where B ∈ ℝ^(d×r), A ∈ ℝ^(r×d)
- r controls the expressiveness of the adaptation

**Trade-offs:**

| Rank | Parameters | Expressiveness | Memory | Training Speed |
|------|-----------|----------------|---------|----------------|
| r=4  | Very few  | Limited        | Minimal | Fastest        |
| r=8  | Few       | Moderate       | Low     | Fast           |
| r=16 | Moderate  | Good           | Medium  | Medium         |
| r=32 | Many      | High           | Higher  | Slower         |
| r=64 | Very many | Very high      | High    | Slowest        |

**Why r=16?**

1. **Empirical sweet spot**:
   - Hu et al. (2021): r=16 matches full fine-tuning on many tasks
   - Further increases (r=32, r=64) give diminishing returns

2. **Capacity analysis**:
   - For Qwen3-0.6B: r=16 → 4.8M trainable parameters
   - r=8: 2.4M params (may underfit complex tasks)
   - r=32: 9.6M params (unnecessary for most tasks)

3. **Insurance underwriting domain**:
   - Specialized vocabulary but not huge
   - Moderate domain shift from pre-training
   - r=16 provides enough capacity to adapt

4. **Memory efficiency**:
   - Still 100x fewer parameters than full fine-tuning
   - Allows larger batch sizes / longer sequences

**Theoretical justification:**
- Hypothesis: weight updates during fine-tuning have low "intrinsic rank"
- Most information in top 16 singular values of ΔW
- Higher ranks capture mostly noise

**When to use different ranks:**
- r=4-8: very similar domains, limited data
- r=16: standard choice, works well broadly
- r=32-64: very different domains, large datasets

**References:**
- Hu et al. (2021), "LoRA", extensive ablation on rank
- Aghajanyan et al. (2020), "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"

---

### 19. LoRA Alpha

**Value:** `lora_alpha = 32` (with `lora_r = 16`)

**Theoretical Background:**

**What is alpha?**
- Scaling factor for LoRA updates
- Actual update: ΔW = (alpha / r) * BA
- Effective learning rate multiplier for LoRA layers

**Formula:**
```python
scaling = lora_alpha / lora_r
ΔW_scaled = scaling * ΔW
```

**Why alpha=32 with r=16?**
- Scaling factor = 32/16 = 2.0
- LoRA updates are scaled 2x compared to base magnitude

**Intuition:**

1. **Prevents under-utilization**:
   - Without scaling: small r → small updates → slow learning
   - Scaling compensates for reduced rank

2. **Rank-independent tuning**:
   - Can change r without retuning learning rate
   - Keep alpha/r ratio constant for similar dynamics

3. **Standard practice**:
   - alpha = 2*r is common convention
   - Provides balanced initialization scale

**Common configurations:**
```
r=8,  alpha=16  → scaling=2.0
r=16, alpha=32  → scaling=2.0
r=32, alpha=64  → scaling=2.0
```

**Effect on training:**
- Higher alpha → stronger LoRA influence
- Lower alpha → more conservative adaptation
- alpha=r → no scaling (scaling=1.0)

**Mathematical perspective:**
- LoRA matrices initialized as:
  - A ~ Gaussian(0, σ)
  - B = 0
- Initial ΔW magnitude ~ σ * sqrt(r)
- Scaling alpha/r normalizes across ranks

**References:**
- Hu et al. (2021), scaling factor discussion in Section 4.1
- Community best practices from HuggingFace PEFT library

---

### 20. LoRA Dropout

**Value:** `lora_dropout = 0.05`

**Theoretical Background:**

**What is LoRA dropout?**
- Dropout applied to LoRA adapter outputs
- During training: randomly zero out LoRA contributions with probability p
- During inference: use full LoRA (no dropout)

**Mechanism:**
```python
h = W_pretrained * x + dropout(LoRA(x), p=0.05)
```

**Why use dropout for LoRA?**

1. **Regularization**:
   - Prevents LoRA adapters from overfitting
   - Forces more robust feature learning
   - Especially important for small datasets

2. **Co-adaptation prevention**:
   - Prevents LoRA from relying on specific activation patterns
   - Encourages diverse adaptation strategies

3. **Small parameter count**:
   - LoRA has few parameters → higher overfitting risk
   - Dropout provides regularization without adding parameters

**Why 0.05 (5%)?**

1. **Low dropout for fine-tuning**:
   - Pre-trained base is already robust
   - Only adapters need regularization
   - Too much dropout can prevent effective adaptation

2. **Comparison:**
   - Training from scratch: 0.1-0.3 dropout
   - Full fine-tuning: 0.1 dropout
   - LoRA: 0.05 dropout (lighter touch)

3. **Empirical validation**:
   - Hu et al. use 0.05-0.1 in experiments
   - 0.05 works across diverse tasks

**Alternative values:**
- 0.0: no regularization (risk overfitting on small data)
- 0.05: standard (balanced)
- 0.1: more aggressive (for very small datasets)

**References:**
- Srivastava et al. (2014), "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
- Hu et al. (2021), dropout in LoRA adapter experiments

---

### 21. LoRA Target Modules

**Values:**
```python
lora_target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",      # Attention
    "gate_proj", "up_proj", "down_proj",         # MLP
]
```

**Theoretical Background:**

**What are target modules?**
- Linear layers in the transformer where LoRA adapters are inserted
- Each module gets its own LoRA decomposition

**Transformer architecture (per layer):**
```
                Input
                  ↓
    ┌─────────────────────────┐
    │  Multi-Head Attention   │
    │   q_proj, k_proj,       │  ← LoRA here
    │   v_proj, o_proj        │
    └─────────────────────────┘
                  ↓
    ┌─────────────────────────┐
    │   Feed-Forward (MLP)    │
    │   gate_proj, up_proj,   │  ← LoRA here
    │   down_proj             │
    └─────────────────────────┘
                  ↓
               Output
```

**Why these specific modules?**

### Attention modules (q_proj, k_proj, v_proj, o_proj):

**q_proj (Query projection):**
- Most critical for task adaptation
- Controls what information the model "asks for"
- Adapting queries changes attention patterns significantly

**k_proj (Key projection):**
- Determines what information is "offered"
- Complementary to queries
- Important for domain-specific entity recognition

**v_proj (Value projection):**
- Controls what information is actually passed forward
- Adapting values changes output representations

**o_proj (Output projection):**
- Final transformation of attention output
- Blends information from multiple heads
- Important for task-specific output formatting

### MLP modules (gate_proj, up_proj, down_proj):

**gate_proj (Gating mechanism):**
- Controls information flow in Gated Linear Units (GLU)
- Qwen uses SwiGLU activation: SwiGLU(x) = Swish(gate_proj(x)) ⊙ up_proj(x)
- Adapting gate changes which features are emphasized

**up_proj (Expansion):**
- Expands hidden_dim → intermediate_dim (usually 4x)
- First stage of feed-forward processing
- Adapting up_proj allows learning domain-specific features

**down_proj (Contraction):**
- Contracts intermediate_dim → hidden_dim
- Final MLP output projection
- Adapting down_proj shapes the final representations

**Why ALL major projections?**

1. **Maximum adaptation capacity**:
   - Each module serves different purpose
   - Adapting all allows comprehensive domain shift

2. **Empirical results (Hu et al. 2021)**:
   - Targeting only attention: works but suboptimal
   - Targeting only MLP: works but suboptimal
   - Targeting both: best performance

3. **Negligible cost**:
   - With r=16, each module adds only ~30K parameters
   - 7 modules × 24 layers = 168 LoRA adapters
   - Still <5M total parameters

**Modules NOT targeted:**

- **Layer norms**: few parameters, not expressive
- **Embeddings**: already task-agnostic
- **LM head**: usually frozen in fine-tuning

**Alternative strategies:**

1. **Attention-only** (q_proj, k_proj, v_proj, o_proj):
   - 43% fewer LoRA parameters
   - Good for minimal adaptation

2. **QV-only** (q_proj, v_proj):
   - Original LoRA paper default
   - 70% fewer parameters
   - Sufficient for similar domains

3. **All-linear** (including layer norms):
   - Marginal gains, not worth cost

**References:**
- Hu et al. (2021), ablation study on target modules (Table 5)
- Qwen technical report for architecture details

---

## Mixed Precision Training

### 22. FP16 vs BF16

**Configuration:**
```python
use_fp16 = True
use_bf16 = False  # T4 doesn't support BF16
```

**Theoretical Background:**

**Floating point formats:**

| Format | Total Bits | Exponent | Mantissa | Range | Precision |
|--------|-----------|----------|----------|-------|-----------|
| FP32   | 32        | 8        | 23       | ±3.4e38 | ~7 decimal digits |
| FP16   | 16        | 5        | 10       | ±65,504 | ~3 decimal digits |
| BF16   | 16        | 8        | 7        | ±3.4e38 | ~2 decimal digits |

**FP16 (IEEE 754 half precision):**
- **Pros**:
  - 50% memory reduction vs FP32
  - 2-3x faster on modern GPUs (Tensor Cores)
  - Supported on all recent NVIDIA GPUs (V100, T4, A100, etc.)

- **Cons**:
  - Limited range: max value 65,504 (overflow → NaN)
  - Gradient underflow for small values (<6e-5)
  - Requires careful loss scaling

**BF16 (Brain Float 16):**
- **Pros**:
  - Same range as FP32 (no overflow issues)
  - No loss scaling needed
  - Simpler training (fewer hyperparameters)

- **Cons**:
  - Less precision (fewer mantissa bits)
  - Only supported on new hardware (A100, H100, AMD MI250)
  - Not available on T4, V100

**Why FP16 for this project?**

1. **Hardware constraint**: T4 GPU doesn't support BF16
2. **Memory savings**: Critical for fitting model + LoRA + optimizer
3. **Speed**: 2x faster than FP32 on T4 Tensor Cores

**Mixed precision training strategy:**

```
Weights:           FP16
Activations:       FP16
Gradients:         FP16
Optimizer states:  FP32  ← critical for stability
Master weights:    FP32  ← high-precision copy
```

**Key techniques for FP16 stability:**

1. **Loss scaling**:
   - Multiply loss by scale factor (e.g., 2^16)
   - Prevents gradient underflow
   - Divide gradients by scale before optimizer step

2. **Gradient clipping**:
   - Prevents gradient overflow (>65,504)
   - Essential in FP16

3. **FP32 accumulation**:
   - Accumulate small updates in FP32
   - Convert back to FP16 for next forward pass

**Why not BF16 when available?**
- BF16 is preferable if hardware supports it
- Simpler training, no loss scaling
- Increasingly standard for new projects
- But FP16 works well with proper precautions

**Performance comparison (T4 GPU):**
```
FP32: 100% memory, 1.0x speed
FP16: 50% memory, 2-3x speed  ← we use this
BF16: not supported on T4
```

**References:**
- Micikevicius et al. (2017), "Mixed Precision Training"
- NVIDIA Apex documentation
- PyTorch Automatic Mixed Precision (AMP) guide

---

## Summary Table

| Hyperparameter | Value | Primary Motivation |
|----------------|-------|-------------------|
| **Memory** |
| Model memory (FP16) | 1.2 GB | 50% reduction vs FP32 |
| LoRA params | 4.8M | 0.77% of full model (619M) |
| Optimizer memory | 38 MB | AdamW: 8 bytes per trainable param |
| Safety margin | 85% | Prevent OOM, allow framework overhead |
| **Batch Size** |
| Physical batch size | 1-4 | Fit in 16GB GPU, cap at 4 for stability |
| Gradient accumulation | Variable | Achieve effective batch of 8 |
| **Training Schedule** |
| Epochs | 3 | Balance adaptation vs overfitting |
| Warmup ratio | 10% | Stabilize early training |
| **Learning Rate** |
| Base LR | 2e-4 | Standard for LoRA fine-tuning |
| LR scaling | sqrt(batch/8) | Conservative scaling for small batches |
| **Optimizer (AdamW)** |
| Beta1 | 0.9 | Momentum over ~10 steps |
| Beta2 | 0.999 | Variance over ~1000 steps |
| Epsilon | 1e-8 | Numerical stability |
| Weight decay | 0.01 | Moderate regularization |
| Grad clip | 1.0 | Prevent exploding gradients |
| **Scheduler** |
| Type | Cosine | Smooth convergence, no tuning needed |
| Min LR ratio | 0.1 | Maintain learning at end |
| **LoRA** |
| Rank (r) | 16 | Sweet spot: capacity vs efficiency |
| Alpha | 32 | 2x scaling factor |
| Dropout | 0.05 | Light regularization |
| Target modules | 7 | All attention + MLP projections |
| **Mixed Precision** |
| FP16 | True | T4 GPU: 2x speed, 50% memory |
| BF16 | False | Not supported on T4 |

---

## References

### Foundational Papers

1. **Transformers & Attention:**
   - Vaswani et al. (2017), "Attention is All You Need"

2. **Optimization:**
   - Kingma & Ba (2014), "Adam: A Method for Stochastic Optimization"
   - Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization" (AdamW)
   - Loshchilov & Hutter (2016), "SGDR: Stochastic Gradient Descent with Warm Restarts"

3. **Learning Rate Scaling:**
   - Goyal et al. (2017), "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
   - Smith et al. (2018), "Don't Decay the Learning Rate, Increase the Batch Size"

4. **Parameter-Efficient Fine-Tuning:**
   - Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models"
   - Aghajanyan et al. (2020), "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"

5. **Mixed Precision Training:**
   - Micikevicius et al. (2017), "Mixed Precision Training"

6. **Regularization & Training Techniques:**
   - Srivastava et al. (2014), "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
   - Pascanu et al. (2013), "On the difficulty of training RNNs"

7. **Memory Optimization:**
   - Chen et al. (2016), "Training Deep Nets with Sublinear Memory Cost"

### Practical Guides

- NVIDIA Deep Learning Performance Guide
- PyTorch Mixed Precision Training Guide
- HuggingFace Transformers Documentation
- HuggingFace PEFT Library Documentation

---

## Appendix: Calculation Example

**Given:**
- Model: Qwen3-0.6B (619M parameters)
- GPU: T4 16GB
- Max sequence length: 512 tokens
- Training examples: 1,000

**Step-by-step calculation:**

1. **Model memory:**
   - 619M × 2 bytes (FP16) = 1,238 MB = 1.21 GB

2. **LoRA parameters:**
   - Rank r=16, d_model=896, 24 layers, 7 modules
   - 2 × 16 × 896 × 24 × 7 = 4,816,896 ≈ 4.8M parameters

3. **Optimizer states:**
   - 4.8M × 8 bytes (AdamW FP32) = 38.4 MB

4. **Activation memory per example:**
   - 512 tokens × 24 layers × 12 bytes = 147 KB

5. **Batch size calculation:**
   - Available memory: (16 - 1.21 - 0.038) × 0.85 = 12.6 GB
   - Memory per example: 147 KB = 0.00014 GB
   - Theoretical batch size: 12.6 / 0.00014 = 90,000
   - **Capped at 4** for training stability

6. **Gradient accumulation:**
   - Target effective batch: 8
   - Physical batch: 4
   - Accumulation steps: 8 / 4 = 2
   - Effective batch: 4 × 2 = 8

7. **Training schedule:**
   - Steps per epoch: 1,000 / 8 = 125
   - Total steps: 125 × 3 = 375
   - Warmup steps: 375 × 0.1 = 38 steps

8. **Learning rate:**
   - Base: 2e-4
   - Scaling: sqrt(8/8) = 1.0
   - Final: 2e-4 × 1.0 = 2e-4

9. **Total memory usage:**
   - Model: 1.21 GB
   - Optimizer: 0.038 GB
   - Activations: 4 × 0.00014 = 0.00056 GB
   - **Total: 1.25 GB (7.8% of 16 GB)**

This conservative estimate ensures stable training with headroom for PyTorch/CUDA overhead.
