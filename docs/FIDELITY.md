# Fidelity and reproducibility boundary

This is an independent implementation of the orchestration described in [EvoTS-Agent v1](https://arxiv.org/html/2608.17933v1), not author code or a claim of matching its tables. No official EvoTS repository URL was present in the paper; the linked detector repository was ruptures.

| Component | This implementation | Reproduction status |
|---|---|---|
| Search control | Two warm-up trials per selected family; incumbent revision; consume stagnant revisions once; final recombination; strict improvement selection | Implements the described control rules |
| Evidence | Plans, scripts, diffs, scores, lineage, logs and API records | Implemented; concise proposal summary replaces a separate post-execution LLM summarization call |
| EDA | Training-only temporal, spectral and adjacent-window statistics; optional unlabeled image | Independent definitions for features not fully specified in the paper |
| PELT / BottomUp / Window | Official ruptures estimators | Library algorithms; independent parameter choices |
| ChangeForest RF / kNN | Official changeforest package | Library algorithms; independent parameter choices |
| Bayesian detector | Exact offline Gaussian/NIG partition inference | Independent implementation; deliberately **offline**, following the paper's model-bank paragraph |
| Spectral detector | Adjacent-window log-power discrepancy | Independent baseline; author's exact spectral code unavailable |
| KL-CPD | GRU generator/critic adaptation with exact multi-RBF MMD | Objective/architecture adaptation; not byte-for-byte original, nor author-exact EvoTS integration |
| Synthetic datasets | Seeded piecewise Gaussian and OU generators | New generated data; author parameter distributions and seeds unspecified |
| Bee-Dance | Six upstream KL-CPD recordings, each split separately | Public upstream data; exact EvoTS aggregation/preprocessing unspecified |
| ADIA | Import plus transparent grouping/stitching, recording and labeling seams | Independent reconstruction; original challenge data not bundled |
| Comparison agents | Fixed detectors, random search, ablations | TS-Agent, DS-Agent and ResearchAgent are **not** impersonated or reproduced |
| LLM-generated implementation | Optional Docker `python` mode | Closest mode to executable-code evolution; local Docker integration must be validated on your machine |
| Restricted search | `spec` mode | Engineering variant; finite configuration space, not equivalent to free code evolution |

Unspecified values include K, optimization budget, stagnation epsilon, full prompts, decoding configuration, preprocessing defaults, exact synthetic sizes, detailed ADIA construction and matching tie rules. Defaults in this repository are explicit engineering choices. Multiple-series metrics use equal-weight macro averaging; record this when comparing other evaluators.

Additional conventions: chronological split indices use floor; a change exactly at a split edge is dropped as an endpoint; both-empty boundaries score F1=1 and Hausdorff=0; one-empty Hausdorff is mathematically infinite and encoded as null with a flag. F1 tolerance is inclusive. No true change count is passed to a detector. Models are refitted on the original training segment for final test prediction, with the selected recipe and seed frozen; validation observations are not added to training.

For stronger research claims, obtain the original scripts/data/protocol, pin provider and model versions, save the Docker image digest, and run enough independently repeated experiments. Search budget equality here means candidate slots, not equal FLOPs, wall time, or tokens; ensembling, repairs and deep training change cost. Reusing the same test set while developing extensions eventually turns it into validation data; reserve a separate final benchmark.

References checked for implementation:

- [Paper v1](https://arxiv.org/html/2608.17933v1)
- [OpenRouter chat completions](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request)
- [OpenRouter errors](https://openrouter.ai/docs/api/reference/errors-and-debugging)
- [OpenRouter structured responses](https://openrouter.ai/docs/guides/features/structured-outputs)
- [ruptures](https://github.com/deepcharles/ruptures)
- [changeforest](https://github.com/mlondschien/changeforest)
- [KL-CPD source](https://github.com/OctoberChang/klcpd_code)
- [ADIA challenge](https://docs.crunchdao.com/competitions/competitions/adia-lab-structural-break-challenge)
