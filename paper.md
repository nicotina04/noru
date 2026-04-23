---
title: "NORU: A zero-dependency Rust library for configurable NNUE training and CPU inference"
tags:
  - Rust
  - NNUE
  - game AI
  - SIMD
  - quantization
authors:
  - name: "Hogyung Choi"
    corresponding: true
    affiliation: "1"
affiliations:
  - index: 1
    name: "Independent Researcher"
date: 23 April 2026
bibliography: paper.bib
---

# Summary

NORU is a pure-Rust software library for building, training, quantizing, and
deploying efficiently updatable neural networks (NNUEs) for game evaluation.
The library combines FP32 training, i16 quantized CPU inference, SIMD kernels,
model serialization, quantization-audit utilities, and a C ABI in one crate.
Unlike engine-specific NNUE implementations, NORU exposes runtime-configurable
network topology through `NnueConfig`, sparse feature-list interfaces, and
incremental accumulator updates so that the same core can be reused across
different game domains without rewriting the neural-network stack.

The motivating use case is non-chess game AI. NNUE has become a proven design
for strong board-game evaluation, but most practical implementations are
embedded inside a single engine and tied to that engine's feature encoding,
search assumptions, and deployment path [@nasu2018nnue; @stockfish2026;
@rapfi2026]. NORU factors out the reusable pieces into a standalone Rust
library that remains light enough for direct inclusion in game-engine code.

# Statement of need

Researchers and engine authors working outside mainstream chess infrastructure
often face an awkward trade-off when adopting NNUE. One option is to fork a
domain-specific engine such as Stockfish or Rapfi and then adapt large amounts
of unrelated search, protocol, and feature-extraction code before reaching the
neural-network parts that are actually needed [@stockfish2026; @rapfi2026].
The other option is to train models in a general-purpose framework such as
PyTorch and then separately build a custom CPU inference path, quantization
pipeline, binary format, and foreign-function interface for deployment
[@paszke2019pytorch].

That split is expensive in small research projects. It creates duplicated model
definitions, weakens reproducibility between training and deployment, and
raises the barrier for experimenting with NNUE in games that are not already
served by a mature engine ecosystem. NORU addresses that gap by packaging the
entire CPU-oriented NNUE workflow into a reusable Rust crate:

- training and inference live in the same codebase;
- topology is configurable at runtime rather than hard-coded per engine;
- serialization and quantization are library features rather than ad hoc
  application glue;
- tree-search-facing operations such as accumulator refresh, incremental
  update, clone, copy, and undo are part of the public API;
- non-Rust hosts can embed the library through a maintained C ABI.

The result is intended for research software authors who need a portable NNUE
core for custom board games, tactical simulations, or other sparse-feature
domains, but who do not want to adopt an entire engine codebase just to obtain
efficient quantized inference.

# State of the field

The NNUE approach originates in computer shogi, where efficiently updatable
evaluation made neural-network guidance practical inside alpha-beta search
[@nasu2018nnue]. The most visible production use is Stockfish, whose NNUE stack
demonstrated that quantized CPU inference and accumulator reuse can deliver
competitive strength in chess without GPU dependence [@stockfish2026;
@stockfishnnue2026]. Similar ideas have also been adopted in Gomoku engines
such as Rapfi [@rapfi2026].

These systems prove the value of NNUE, but they are not general-purpose
research libraries. Their implementations are coupled to one game's feature
schema and to the surrounding engine architecture. At the other end of the
spectrum, machine-learning frameworks such as PyTorch provide flexible training
tooling but intentionally leave deployment-specific concerns, CPU quantization,
incremental state maintenance, and engine integration to the application layer
[@paszke2019pytorch].

NORU occupies the middle ground. It is narrower than a full ML framework and
broader than a single-engine implementation. The library deliberately focuses
on the part that is reusable across domains: sparse-feature NNUE definition,
incremental accumulator management, training support, quantized CPU inference,
and reproducible weight serialization.

# Software design

NORU is organized as a small set of public modules that map onto the typical
NNUE lifecycle.

The `noru::config` module defines `NnueConfig`, which can either borrow a
compile-time hidden-layer layout or own a runtime one. This allows the same
library surface to support both fixed embedded engines and FFI-created models.
The runtime-config ownership path is intentionally safe: topology can now be
owned directly rather than leaked and manually reclaimed.

The `noru::trainer` module provides FP32 trainable weights, forward passes with
intermediate activations, binary cross-entropy and raw-output MSE backprop,
Adam optimization, and self-describing FP32 checkpoint serialization. The
`noru::network` module provides the deployment representation: i16 quantized
weights, versioned binary load/save, accumulator refresh and incremental
update, and integer forward evaluation.

The accumulator is dual-perspective by design: side-to-move and opponent
feature sums are maintained separately and then concatenated before the hidden
layers. This matches the common NNUE pattern for search engines while keeping
the feature extractor game-agnostic. Hidden layer topology is runtime sized,
and inference kernels are specialized through AVX2, NEON, and scalar fallback
paths.

Recent library additions also expose quantization drift as a first-class API.
The `noru::audit` module reports sign agreement, output-range statistics,
inferred output scaling, and probability-space error between FP32 and i16
models, so deployment quality can be measured without writing custom harnesses.

Finally, `noru::ffi` exposes the same capabilities to non-Rust hosts. The C
ABI includes trainer lifecycle, gradient updates, quantization, weight load and
save, accumulator creation, refresh, incremental update, undo, clone, and
forward evaluation. This keeps training and inference behavior aligned across
Rust and external engine embeddings.

# Research impact statement

NORU's primary contribution is enabling cross-domain NNUE experimentation with
a small, inspectable, dependency-free codebase. Within the author's current
projects, the library already serves as the shared NNUE core for a public
Gomoku engine integration (`figrid-board`) [@figrid2026]. In the same local
workspace family, the author also uses NORU in a hex-grid tactical battler and
in a Connect 4 ablation target. The repository-level benchmark inventory makes
those distinctions explicit so that public downstream evidence is separated
from internal or not-yet-packaged artifacts.

To keep those claims auditable, this repository now includes a benchmark
inventory that separates currently public artifacts from internal or
not-yet-packaged ones. The public Gomoku integration is already reproducible at
the repository level; the other domains are listed together with the concrete
benchmark artifacts that still need to be externalized before final
publication. That separation matters for JOSS review because it distinguishes
implemented capability from public reproducibility status instead of conflating
the two.

In practical terms, NORU lowers the cost of answering research questions such
as:

- how much of NNUE's benefit comes from the architecture rather than the game;
- whether one sparse-feature evaluation stack can serve multiple domains;
- how much quantization error a deployment pipeline introduces;
- what extra search support a non-chess engine needs once a reusable NNUE core
  is available.

Those are software questions as much as modeling questions, and they are best
served by a library that keeps the training, serialization, quantization, and
CPU inference path in one place.

# AI usage disclosure

Generative AI tools were used during the development and maintenance of this
repository to brainstorm implementation approaches, draft issue and pull
request text, review code changes, and help draft documentation. All AI-assisted
changes were manually reviewed by the author and were merged only after local
inspection and automated verification. No empirical claim, benchmark summary,
or citation in this manuscript is intended to rely on unverified AI output.

# Acknowledgements

No dedicated research funding supported this work. The author thanks the open
source NNUE communities around shogi, Stockfish, and Gomoku for making the
design space legible enough that a reusable library implementation became
practical.
