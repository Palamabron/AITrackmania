# Experiment orchestrator data

Versioned in git:

- `configs/` — per-experiment Hydra override YAMLs
- `search_space.yaml`, `baseline.yaml`, `orchestrator_config.yaml`
- `audit_decisions.py`, `extract_ground_truth.py`
- `reset-*.cmd`

Generated locally (gitignored):

- `analysis/*.json` — W&B metric exports (`scripts/fetch_analysis.py`)
- `registry.jsonl` — run registry (`tmrl.tools.orchestrator`)
- `logs/` — orchestrator logs
- `ground_truth.json`, `audit_report.json`, `validation_report.json`, `decisions.md`

After cloning, run experiments via `make orchestrator` or `uv run python -m tmrl.tools.experiment_manager`; artifacts appear under the paths above.
