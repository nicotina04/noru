# Contributing to NORU

NORU is developed in the open. Small, reviewable changes are preferred over large mixed commits.

## Ways to contribute

- Report bugs with a minimal reproducer.
- Propose API improvements with a concrete use case.
- Improve examples, tests, and documentation.
- Benchmark behavior across games or platforms and share results.

## Before opening a change

1. Open an issue that states the problem, expected behavior, and affected API.
2. Keep each pull request focused on one topic.
3. Add or update tests when behavior changes.
4. Update README or rustdoc when the public API changes.

## Local verification

Run the same checks expected in CI:

```bash
cargo fmt --check
cargo test
cargo doc --no-deps
cargo package --allow-dirty --list
```

## Issue template guidance

Useful bug reports include:

- `noru` version
- target platform and Rust version
- network configuration
- sample feature lists or serialized weights if relevant
- expected behavior vs actual behavior

Useful feature requests include:

- the research or engine workflow blocked by the missing feature
- why the feature belongs in `noru` rather than an application crate
- proposed API shape if you already have one

## Scope

NORU aims to stay:

- dependency-free
- game-agnostic
- usable from both pure Rust and FFI hosts
- conservative about unsafe code and serialization compatibility

Changes that add substantial complexity should explain the tradeoff in the issue or PR description.
