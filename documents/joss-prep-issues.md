# JOSS Prep Issues for NORU

This list turns the current JOSS preparation work into concrete, reviewable issue-sized tasks.

## High Priority

### 1. Add continuous integration for public verification

- Problem: the repository has tests and docs locally, but no public CI signal.
- Deliverables:
  - GitHub Actions workflow for `cargo fmt --check`
  - `cargo test`
  - `cargo doc --no-deps`
  - `cargo package --allow-dirty --list`
- Acceptance criteria:
  - workflow passes on `main`
  - README development section matches CI commands

### 2. Add contribution, conduct, and citation metadata

- Problem: JOSS reviewers expect clear support and contribution pathways.
- Deliverables:
  - `CONTRIBUTING.md`
  - `CODE_OF_CONDUCT.md`
  - `CITATION.cff`
- Acceptance criteria:
  - repository documents how to report issues and propose changes
  - citation metadata is valid YAML and points at the repository

### 3. Make `FeatureDelta` construction checked instead of silent-truncation by default

- Problem: the public Rust API silently drops entries beyond capacity, which is dangerous for library users.
- Deliverables:
  - checked constructor from slices
  - checked per-entry append methods
  - tests for overflow behavior
  - FFI path uses the checked constructor
- Acceptance criteria:
  - overflow is reported explicitly in the checked API
  - FFI continues to return `NORU_ERR_INVALID_ARG`

## Medium Priority

### 4. Reduce unsafe configuration ownership

- Problem: runtime-configured topologies currently rely on leaking `hidden_sizes` and reclaiming them manually.
- Deliverables:
  - design note comparing current model vs safer ownership alternatives
  - implementation that removes or narrows `reclaim_leaked_hidden_sizes`
- Acceptance criteria:
  - public API requires less manual memory ownership reasoning
  - FFI no longer needs paired reclaim logic in `Drop`

### 5. Add a quantization audit/report API

- Problem: quantization quality can be demonstrated today only through examples and ad hoc checks.
- Deliverables:
  - reusable API that compares FP32 vs i16 outputs on a sample set
  - metrics such as sign agreement and error summary
  - example or benchmark documentation
- Acceptance criteria:
  - users can quantify deployment drift without writing custom harnesses
  - JOSS paper can cite reproducible library-level metrics

### 6. Add a paper scaffold and benchmark inventory

- Problem: repository documentation is strong, but JOSS paper materials are still missing.
- Deliverables:
  - `paper.md`
  - `paper.bib`
  - benchmark table inputs for Gomoku / hex battler / Connect 4
- Acceptance criteria:
  - paper includes statement of need, state of the field, software design, and research impact

## Lower Priority

### 7. Add external reproducibility examples

- Problem: current examples are toy-scale and API-focused.
- Deliverables:
  - one example showing a realistic feature extractor loop
  - one example showing FFI embedding from a non-Rust host
- Acceptance criteria:
  - examples help a reviewer understand real usage, not just toy training

### 8. Collect public adoption evidence

- Problem: JOSS review weighs research impact and credible reuse.
- Deliverables:
  - list of downstream repositories or engines using NORU
  - benchmark or writeup showing why NORU exists instead of extending another library
- Acceptance criteria:
  - the paper can point to concrete reuse or reproducible capability evidence
