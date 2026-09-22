# Included examples

- `smoke-inputs/`: generated mean/variance data, chronological splits. Two 1500-sample series, seed 42, dimension 1, regular segment length 100. Training labels have been removed.
- `smoke-run/`: actual 8-trial infrastructure run with `configs/smoke.json`. **No language model was used.** Open `report.html` alongside its `search.png`.
- `bee-pelt-baseline/`: frozen fixed-PELT baseline plus measured test output on six public Bee-Dance recordings. **This is not an EvoTS LLM run.** Dataset bytes are acquired with `fetch-bee` and are not included here.
- `adia_manifest.example.json`: input schema, with illustrative filenames to replace locally.

To reproduce the included synthetic workflow without overwriting its recorded output:

```bash
python -m evots run --prepared examples/smoke-inputs --config configs/smoke.json --output runs/new-smoke
python -m evots evaluate --run runs/new-smoke
python -m evots report --run runs/new-smoke
```

Recorded manifests contain the original absolute prepared-data path for provenance. For a relocated run, `evaluate --prepared YOUR_PATH` validates the same dataset hashes without editing the manifest. Run resume requires identical code and scientific configuration.
