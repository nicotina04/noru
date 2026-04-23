# Adoption Evidence

This document records what can currently be claimed, with evidence, about NORU
adoption and community-readiness. It intentionally separates verified public
downstream use from local workspace use and from future-facing aspirations.

## Verified Public Downstream

### `figrid-board`

Verified on 2026-04-23 via `gh repo view nicotina04/figrid-board`.

- Repository: <https://github.com/nicotina04/figrid-board>
- Visibility: public
- Description: "A five-in-a-row (gomoku) library written in Rust."
- Current role: public Gomoku engine / library downstream using NORU as the
  NNUE core

Local supporting evidence in the current workspace:

- `/mnt/c/Users/concreate/Documents/workspace/rust/figrid/Cargo.toml`
- `/mnt/c/Users/concreate/Documents/workspace/rust/figrid/bin/pbrain_figrid_noru.rs`
- `/mnt/c/Users/concreate/Documents/workspace/rust/figrid/CHANGELOG.md`

What can be claimed safely:

- NORU already has at least one verified public downstream repository.
- That downstream is not a toy example; it is a Gomoku engine integration with
  embedded NNUE weights and a tournament-facing protocol binary.

## In-Workspace Known Consumers

These are real consumers in the author's local workspace, but they should not
be described as public external adoption unless their repository status is
verified separately.

### `noru-tactic`

Observed local crates depending on NORU:

- `crates/gomoku-engine`
- `crates/gomoku-trainer`
- `crates/hex-battle`
- `crates/hex-trainer`

What this supports:

- the library is already being reused across multiple domains and binaries;
- NORU is not only exercised by its own examples and tests;
- the FFI/inference/training APIs are used in larger application codebases.

What this does *not* support by itself:

- a claim of public third-party adoption;
- a claim of external scholarly reuse.

## Repository-Level Community Readiness Signals

These are not adoption by themselves, but they strengthen the case for
credible near-term significance in a JOSS submission.

- public CI workflow for formatting, tests, docs, and package contents
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `CITATION.cff`
- C ABI surface in `src/ffi.rs`
- runnable Rust examples in `examples/`
- software-paper scaffold in `paper.md`
- benchmark inventory in `documents/benchmark_inventory.md`

## Claims To Avoid Until More Evidence Exists

The following claims should remain out of the paper unless new evidence is
added:

- "widely adopted"
- "used by multiple public downstream projects" beyond `figrid-board`
- "enabled published research" unless concrete papers are listed
- specific Connect 4 or hex battler benchmark claims without public artifacts

## Recommended Next Evidence To Collect

1. externalize one benchmark artifact from the Gomoku downstream integration
2. package one non-Rust embedding example around `examples/ffi_embed.c`
3. verify whether `noru-tactic` or another consumer is public before citing it
4. look for reverse dependencies or public repositories that mention `noru`
