# AITrackmania — Copilot / coding agent

## Do not use as context (experiment data only)

These paths are generated run artifacts, not source code. **Do not read, summarize, or base reviews on their contents:**

- `experiments/analysis/` (W&B export JSON)
- `experiments/logs/`
- `experiments/registry.jsonl`
- `experiments/ground_truth.json`, `experiments/audit_report.json`, `experiments/validation_report.json`
- `experiments/decisions.md`
- `output_files/`, `wandb/`, `reports/`

If a task does not require experiment metrics, ignore the files above entirely.

## Where to look instead

- Training / env / reward logic: `tmrl/`
- Experiment **configs** (small YAML): `experiments/configs/`, `experiments/search_space.yaml`, `experiments/orchestrator_config.yaml`, `experiments/baseline.yaml`
- Orchestrator code: `tmrl/tools/orchestrator.py`, `tmrl/tools/experiment_manager.py`
- Tests: `tests/`
