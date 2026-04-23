# Benchmark Inventory

This document is the input table for the software-paper benchmark section. Its
goal is not to overstate what is already public; it separates current evidence
from artifacts that still need to be packaged before submission.

## Summary Table

| Domain | Current NORU consumer | Public artifact today | NORU configuration | Current evidence | Reproducibility status |
|--------|------------------------|-----------------------|--------------------|------------------|------------------------|
| Gomoku (15x15 freestyle) | `figrid-board` / `pbrain-figrid` | Yes | feature size 4096, accumulator 512, hidden `[64]`, CReLU | public downstream engine, embedded NNUE weights, Gomocup-oriented binary, changelog-documented integration | Ready for paper with command-level benchmark packaging |
| Hex-grid tactical battler | `noru-tactic` `hex-battle` + `hex-trainer` | Not yet externalized as a standalone public benchmark artifact | feature size 138, accumulator 256, hidden `[64]`, CReLU | workspace integration demonstrates non-board-specific deployment with the same NORU core | Needs public benchmark harness and result snapshot |
| Connect 4 | internal ablation target referenced in README | No public artifact found in current workspace | not yet pinned in repository-level materials | README claim only: reaches about 45% win rate against a depth-matched heuristic after a few hours of training | Needs externalized code, config, and benchmark logs before final submission |

## Evidence Notes

### 1. Gomoku

Current evidence comes from the public `figrid-board` downstream engine in the
same workspace family. The strongest concrete artifact is the `0.4.0` changelog
entry, which documents that NORU-backed NNUE evaluation was integrated into a
Gomocup-targeted binary, with embedded weights and a fixed 4096 -> 512 -> 64 ->
1 configuration.

Relevant local evidence:

- `/mnt/c/Users/concreate/Documents/workspace/rust/figrid/CHANGELOG.md`
- `/mnt/c/Users/concreate/Documents/workspace/rust/figrid/bin/pbrain_figrid_noru.rs`
- `/mnt/c/Users/concreate/Documents/workspace/rust/figrid/src/features.rs`

Paper-safe claim:

- NORU already powers a public Gomoku engine artifact, so the library has at
  least one downstream integration that reviewers can inspect independently.

Recommended benchmark package before submission:

- fixed hardware description
- build command for `figrid-board`
- benchmark suite of opening positions
- opponent set and time controls
- win-rate or move-agreement table against at least one baseline

### 2. Hex-grid Tactical Battler

The hex battler evidence currently lives in the `noru-tactic` workspace. Its
feature extractor is explicitly position-independent and uses a 138-feature
layout with `NnueConfig { feature_size = 138, accumulator_size = 256,
hidden_sizes = [64] }`.

Relevant local evidence:

- `/mnt/c/Users/concreate/Documents/workspace/noru-tactic/crates/hex-battle/src/features.rs`
- `/mnt/c/Users/concreate/Documents/workspace/noru-tactic/crates/hex-trainer/src/pipeline.rs`

Paper-safe claim:

- NORU has already been applied to a non-chess, non-Gomoku domain with a very
  different feature schema, which supports the claim that the runtime-configured
  API generalizes beyond one board game.

Gap before final submission:

- publish or package a reproducible benchmark harness
- define a public metric such as position-ranking accuracy, arena win rate, or
  placement-quality correlation against a heuristic or teacher

### 3. Connect 4

The current repository README states that a minimal Connect 4 target reached
roughly 45% win rate against a depth-matched heuristic after a few hours of
training. However, a corresponding public artifact was not found in the current
workspace scan.

Paper-safe claim:

- Connect 4 should remain described as an internal ablation target unless the
  code, configuration, and result logs are packaged publicly before submission.

Gap before final submission:

- publish the minimal training/inference harness
- pin the exact feature extractor and `NnueConfig`
- record benchmark command lines and opponent settings

## Minimum Reproducibility Package For The Paper

Before final JOSS submission, at least one benchmark row should be fully
reproducible from repository materials. The Gomoku downstream integration is
the most mature candidate and should be treated as the primary artifact.

Recommended checklist:

1. choose one public benchmark harness as the canonical paper artifact
2. record exact commands, hardware, model revision, and dataset or opening set
3. store result tables in-repo rather than only in changelogs or chat notes
4. clearly label other rows as internal, pending, or future reproducibility
