# Extending the experiment

## Add a detector

1. Add its ID and description in `evots/bank.py:BANK`.
2. Implement `detect(train, x, spec, seed)` dispatch, using training observations to fit learned transformations.
3. Declare and bound new hyperparameters in `DEFAULTS` and `validate_spec`.
4. Add optional dependency detection in `available_bank`, dependencies in `pyproject.toml`, and installation in `Dockerfile` where needed.
5. Add a meaningful synthetic test for the type of change it targets and a test that unseen data does not affect fitted statistics.
6. Add the ID to the experiment config. Use a new output directory; source changes deliberately invalidate resume.

`python` mode can already create new algorithms using the installed numerical stack and the `custom` family. Its generated code receives the same contract and is isolated from scoring labels.

## Add a search operator

`runner.choose_operator` controls scheduling. `proposer.Proposer.propose` supplies the operator instruction. Preserve the strict incumbent rule and record parent IDs, failures and cost. Recombination should integrate useful components, not simply pick whichever test score is larger. Keep final test scores out of planning contexts.

## Concrete extension experiments

| Research question | Implementation point | Required comparison |
|---|---|---|
| Does an exploration schedule beat the fixed stagnation trigger? | `choose_operator` | Same candidate and token budgets; multiple seeds |
| How much does feature-only EDA lose versus vision? | `agent.vision` | Same model/provider and selected data |
| Does the learned kernel justify its runtime? | `klcpd_scores` / bank subset | Validation/test F1, Hausdorff, training seconds and cost |
| Can a label-efficient score guide search? | Separate scorer and protocol | Reduce validation annotations in advance; keep full held-out test fixed |
| Can a selected detector transfer to another asset? | New dataset pairing protocol | Search on source validation, freeze, then evaluate target |
| What changes for online detection? | New causal detector/evaluator | Detection delay, causal windows, false alarms; never relabel offline results as online |

These are proposed experiments, not demonstrated improvements. For label-efficient experiments, remove labels from the scorer as well as from prompts; hiding labels from the LLM alone does not make the optimization unsupervised.

## Reproducibility records

Keep the run manifest, prepared input manifests, frozen candidate, API responses and dependency lock or container digest. If moving runs to another machine, supply `evaluate --prepared NEW_PATH`; all data fingerprints must still match. Do not alter dataset hashes. Do not mix mock runs, restricted spec runs and generated-code runs in the same statistical comparison.

## Failure diagnosis

- 401/402/403 abort immediately: inspect key permissions/credit or provider rejection; retries cannot fix credentials.
- 429/temporary 5xx use bounded retry respecting Retry-After. A request timeout can still incur billing.
- Truncated JSON is recorded as a failed completion. Increase output budget or constrain reasoning in a new experiment; do not treat it as a valid pipeline.
- Failed detector attempts retain logs and candidate code. Repairs receive the error, not test feedback.
- Docker is required for free Python. Missing Docker has a clear error; there is no unsafe local execution fallback.
- `changeforest` kNN can legitimately return no break on a short sequence. A successful execution is not evidence of accurate detection.
- Bayesian offline segmentation has quadratic cost and an explicit 4000-sample limit per evaluated series.
