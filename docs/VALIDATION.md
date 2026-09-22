# Delivery validation

Validated on 2026-09-22, Python 3.12.14, Linux, CPU.

**37 tests passed in 25.45 seconds.** No skips in this environment.

Covered:

- One-to-one maximum-cardinality boundary matching, inclusive tolerance, duplicate/invalid indices and empty-set conventions.
- Chronological split boundaries, removal of training labels and training-only fitted preprocessing.
- Strict incumbent preservation, accepted small gains that still trigger stagnation, one-time use of stagnant evidence and final-step recombination precedence.
- Search with all held-out test arrays physically absent; search still completes and evaluation correctly requires the missing arrays.
- Every detector family, including the installed changeforest variants and the modern KL-CPD adaptation; deterministic CPU KL-CPD repeatability on a small fixture.
- OpenRouter mocked HTTP: 429 retry, Retry-After, authentication failure without retry, errors inside HTTP 200, truncated completion accounting, caching and request limits.
- End-to-end OpenRouter proposal parsing via a mock HTTP transport, real detector execution, freezing and held-out scoring.
- Multi-seed ablation matrix, result CSV and paired comparison.
- ADIA manifest import/stitching and explicit artificial-join labels.
- Missing-Docker handling: no fallback to unrestricted host Python execution.

Additional executed checks:

| Check | Result |
|---|---|
| Source-package wheel build | Successful: `evots_openrouter-0.1.0-py3-none-any.whl` |
| Synthetic mock workflow | 8/8 valid trials; initial/revision/alternative/recombination paths executed |
| Synthetic held-out scoring | Completed; F1=1.0 on this deliberately simple regular-segment fixture |
| HTML/PNG report | Generated from the mock workflow |
| Bee-Dance acquisition | All six pinned upstream files downloaded and parsed |
| Real Bee-Dance PELT baseline | Test macro F1=0.560053, precision=0.454299, recall=0.750000, mean Hausdorff=23.166667 samples |

The Bee-Dance result is one fixed-detector baseline using this repository's protocol. It is not an EvoTS/OpenRouter result and is not directly comparable to a paper table without matching its data preparation and scoring details. Files are in `examples/bee-pelt-baseline/`. The raw upstream dataset is not redistributed in the bundle; use `fetch-bee`.

Not executed in this environment:

- A billed/live OpenRouter generation request; the transport and planner were tested with controlled API responses.
- Building/running Docker: no Docker executable was available here. The container path, limits and entry point are supplied for local integration testing.
- Full paper-scale experiments, the original authors' datasets/configurations, original comparator agents, or GPU training.

Mock scores and unit tests demonstrate implementation behavior, not the paper's empirical claims or scientific superiority. See `FIDELITY.md` before designing a claimed replication.
