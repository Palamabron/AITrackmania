# Release Readiness Design — AITrackmania

**Date:** 2026-07-09  
**Branch:** feat/sota-professional-library  
**Approach:** Severity-first sweep with per-category commits

## Goal

Make the codebase release-ready as a professional RL library for Trackmania. Covers two functional bugs and three cosmetic categories. Excluded scope is listed explicitly.

---

## Commit Plan

### Commit 1 — `fix: ship py.typed marker in wheel (PEP 561)`

**File:** `pyproject.toml`

Add `"py.typed"` to `[tool.setuptools.package-data]` under the `tmrl` key. Without this, the PEP 561 marker file exists on disk but is not included in the built wheel/sdist, so downstream `mypy` users won't see tmrl as a typed package.

```toml
[tool.setuptools.package-data]
tmrl = ["py.typed", "config/defaults/**/*.yaml"]
```

**Verification:** `python -m build --wheel && unzip -l dist/*.whl | grep py.typed`

---

### Commit 2 — `fix: replace hardcoded algo whitelist with registry lookup`

**File:** `tmrl/config/config_objects.py` ~line 74

Replace:
```python
if ALG_NAME not in ("SAC", "REDQSAC", "TQC", "IQN", "SDSAC"):
    raise ValueError(...)
```
With:
```python
if ALG_NAME not in ALGORITHMS.keys():
    raise ValueError(...)
```

This makes the plugin system functional as documented in `CONTRIBUTING.md`. Any algorithm registered via entry points will be accepted at startup instead of being blocked by the hardcoded tuple.

**Verification:** `make check` passes; plugin-registered algo name no longer raises ValueError.

---

### Commit 3 — `docs: add Google-format docstrings to public API gaps`

Add Google-format docstrings (Args/Returns/Raises sections) to all public symbols currently missing them:

| File | Symbols to document |
|---|---|
| `tmrl/util.py` | `pandas_dict`, `shallow_copy`, `get_class_or_function`, `partial_from_args`, `get_output`, `dump`, `load`, `save_json`, `load_json`, `prod` |
| `tmrl/wrappers.py` | `AffineObservationWrapper` class + `__init__`, `observation`, `float64_to_float32`, `float_to_float32`, `int_to_float32` |
| `tmrl/actor.py` | `act_`, `TorchActorModule.save`, `.load`, `.save_to_bytes`, `.to`, `.to_device` |
| `tmrl/envs.py` | `GenericGymEnv` class-level docstring |
| `tmrl/memory/base.py` | `sample`, `append`, `sample_indices` |
| `tmrl/networking/trainer.py` | 2 undocumented public defs |

Pydantic field descriptions in `tmrl/config/schema/` are already complete — no changes needed there.

**Format standard:** Google style (matching existing docstrings in `tmrl/actor.py` and `tmrl/registry.py`).

---

### Commit 4 — `refactor: remove dead code and clean up inline comments`

| File | Change |
|---|---|
| `tmrl/envs.py` lines 32–33 | Remove commented-out `assert` and `NormalizeActionWrapper` lines |
| `tmrl/envs.py` lines 37–38 | Remove empty `if __name__ == "__main__": pass` guard |
| `tmrl/actor.py` line 147 | Remove `# super().__init__()  # torch.nn.Module` leftover comment |
| `tmrl/custom/utils/nn_distributions.py` line 85 | Remove commented-out alternative distribution construction |
| `tmrl/custom/tm/utils/control/gamepad.py` line 24 | Remove commented-out scaling formula |
| `tmrl/tools/diagnostics/check_environment.py` lines 30–31 | Convert `print("d: ", d)` / `print("t: ", t)` to `logger.debug()` |
| `tmrl/tuto/tuto_minimal_drone.py` line 145 | Remove commented-out alternate config example |

**Kept intentionally:**
- `window.py` FIXME and `keyboard.py` TODO — document real known limitations
- `impala_actor_critic.py` `# noqa: F401` re-export shim — intentional

---

### Commit 5 — `chore: fix pyproject.toml metadata gaps`

- Move `pyinstrument` from `[project.dependencies]` to `[project.optional-dependencies] dev`
- Delete `.readthedocs.yaml` (stale v1 format; `.readthedocs.yml` v2 is canonical)

---

## Explicitly Out of Scope

| Item | Reason |
|---|---|
| Restoring 24 deleted test files | Separate initiative; each requires domain knowledge |
| Python 3.13 CI matrix job | Needs validation effort beyond cosmetic work |
| Sphinx `.rst` coverage for `tmrl.registry`, `.models`, etc. | Significant docs infrastructure work |
| LICENSE copyright alignment | Legal/attribution decision for maintainer |
| `tmrl.config.__init__` `__all__` hygiene | Touches many downstream imports; semver concern |
