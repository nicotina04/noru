# NORU NNUE Study Guide

This guide walks you through the noru codebase step-by-step to understand how NNUE works.
Copy each step into Claude web (or any LLM) along with the referenced code.

Repository: https://github.com/nicotina04/noru

---

## Step 1: The Big Picture — What is NNUE?

**Read**: `src/config.rs` (entire file, ~60 lines)

**Ask**:
> Here is my NNUE config struct. Can you explain with a diagram:
> 1. What does each field control in the neural network?
> 2. What does the network look like when `hidden_sizes: &[64]` vs `&[256, 32, 32]`?
> 3. Why are there two perspectives (STM/NSTM) and what does `concat_size()` mean?
> 4. What's the difference between CReLU and SCReLU, and why does SCReLU only apply to the first layer?

**Goal**: Understand the network shape before seeing any math.

---

## Step 2: Sparse Features — How Games Become Numbers

**Read**: Nothing in noru (features are game-specific). Instead, think about this:

**Ask**:
> In NNUE, the input is a list of "active feature indices" like `[0, 42, 100]` out of 530 possible features.
> 1. Why sparse indices instead of a dense vector of 530 floats?
> 2. In chess, features might be "white pawn on e4" = index 203. How would you design features for a 15×15 gomoku board?
> 3. Why does NNUE have separate feature lists for STM (side-to-move) and NSTM (opponent)?

**Goal**: Understand why NNUE uses sparse input and how games map to feature indices.

---

## Step 3: The Accumulator — NNUE's Key Innovation

**Read**: `src/network.rs` — focus on:
- `struct Accumulator` and `fn new()`
- `fn refresh()`
- `fn update_incremental()`
- `struct FeatureDelta`

**Ask**:
> Here is my Accumulator code. Can you explain with a diagram:
> 1. What does `refresh()` actually compute? Walk through the math for a small example (3 features, accumulator_size=4).
> 2. Why is `update_incremental()` faster than `refresh()`? Show the difference when one piece moves (1 feature added, 1 removed).
> 3. Why is this critical for game tree search (alpha-beta)?
> 4. What does `swap()` do and when would you use it?

**Goal**: Understand incremental updates — this is what makes NNUE special vs regular neural nets.

---

## Step 4: Forward Pass — From Accumulator to Evaluation

**Read**: `src/network.rs` — `fn forward()`

**Ask**:
> Here is my quantized (i16) forward pass. Can you explain step by step with numerical examples:
> 1. What does ClippedReLU do and why clamp to [0, 127]?
> 2. Walk through the hidden layer dot product for one output neuron.
> 3. What's ACTIVATION_SCALE (256) and OUTPUT_SCALE (16) — why do we need these scaling factors in integer arithmetic?
> 4. How does the SCReLU path differ? Why does it use i64 instead of i32?
> 5. What does the final output number mean? (centipawns? win probability?)

**Goal**: Understand quantized integer inference and why it's fast.

---

## Step 5: Training — Learning the Weights

**Read**: `src/trainer.rs` — focus on:
- `struct TrainableWeights`
- `fn forward()` (the f32 version)
- `fn backward()` and `fn backward_inner()`

**Ask**:
> Here is my FP32 training code. Compare it to the i16 inference code from Step 4:
> 1. Why train in f32 but infer in i16?
> 2. Walk through `forward()` — how does the sigmoid at the end turn the output into a probability?
> 3. In `backward_inner()`, what is `d_output = fwd.sigmoid - sample.target` doing? (Hint: BCE loss derivative)
> 4. How do gradients flow backwards through the ClippedReLU? What happens when the activation is clamped (gradient = 0)?
> 5. How is the SCReLU backward different? (derivative of x² = 2x)

**Goal**: Understand backpropagation in the context of NNUE.

---

## Step 6: Adam Optimizer — How Weights Get Updated

**Read**: `src/trainer.rs` — `fn adam_update()` and `fn adam_step()`

**Ask**:
> Here is my Adam optimizer. Can you explain:
> 1. What are `m` (momentum) and `v` (velocity) tracking?
> 2. What does bias correction (`bc1`, `bc2`) fix?
> 3. Why is Adam better than plain SGD for NNUE training?
> 4. Why does the feature weight update skip rows where all gradients are zero? (sparse optimization)

**Goal**: Understand how training actually modifies the weights.

---

## Step 7: Quantization — FP32 to i16

**Read**: `src/trainer.rs` — `fn quantize()` and `src/quant.rs`

**Ask**:
> Here is my quantization code (FP32 → i16) and the scaling constants.
> 1. Why multiply by WEIGHT_SCALE (64) when converting?
> 2. What precision do we lose? If a weight is 0.015, what does it become?
> 3. Why is the hidden weight layout transposed during quantization (input-major → output-major)?
> 4. How do ACTIVATION_SCALE and OUTPUT_SCALE compensate during inference to get the right answer despite integer rounding?

**Goal**: Understand the bridge between training and inference.

---

## Step 8: SIMD — Making It Fast

**Read**: `src/simd/scalar.rs` (reference) then `src/simd/avx2.rs`

**Ask**:
> Here is my scalar dot product and the AVX2 version. Can you explain with a diagram:
> 1. What does `_mm256_madd_epi16` do exactly? Show the data flow for 16 input elements.
> 2. Why is the horizontal sum at the end needed?
> 3. For SCReLU, why do we need to widen to i64? Show the overflow math.
> 4. How does the NEON version differ from AVX2?

**Goal**: Understand why SIMD matters for NNUE inference speed.

---

## Step 9: Putting It All Together

**Ask** (no code needed):
> Now I understand all the pieces. Can you draw a complete diagram showing:
> 1. Training pipeline: samples → forward → backward → adam → quantize → save
> 2. Inference pipeline: load → accumulator refresh → forward → eval score
> 3. Search integration: how alpha-beta search uses incremental accumulator updates
> 4. Where SIMD accelerates the inference pipeline

**Goal**: See the full picture of how everything connects.

---

## Bonus: Compare with Stockfish

**Ask**:
> My NNUE library (noru) has:
> - Configurable hidden_sizes (e.g. &[256, 32, 32])
> - CReLU + SCReLU
> - Dual perspective (STM/NSTM)
> - SIMD (AVX2/NEON)
>
> How does this compare to Stockfish's NNUE?
> What does Stockfish have that noru doesn't? (HalfKAv2, king buckets, etc.)
> What would I need to add to use noru for a competitive chess engine?
