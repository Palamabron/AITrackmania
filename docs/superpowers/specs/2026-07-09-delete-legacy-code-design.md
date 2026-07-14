# Delete Legacy Code Design ? AITrackmania

**Date:** 2026-07-09
**Branch:** feat/sota-professional-library
**Approach:** Four targeted commits, each deleting one category of dead/deprecated code.

## Goal

Remove code that exists purely as dead weight or for backward compatibility with pre-v0.9.0 import paths. No public API changes ? the canonical public namespaces (`tmrl.memories`, `tmrl.interfaces`, etc.) are preserved.

---

## Commit Plan

### Commit 1 ? `refactor: remove deprecated custom_algorithms package`

**Delete:** `tmrl/custom/custom_algorithms/__init__.py` (and directory)

This package was renamed to `tmrl.custom.algorithms` in v0.9.0. It fires `DeprecationWarning` at import, contains only re-exports from `tmrl.custom.algorithms`, and has zero internal consumers anywhere in the codebase. Its removal eliminates the only code-level shim between the old and new algorithm package path.

**Verification:** `grep -r "custom_algorithms" tmrl/` returns nothing after deletion.

---

### Commit 2 ? `refactor: delete zero-byte config stubs`

**Delete:**
- `tmrl/config/_internal/_config_agent.py` (0 bytes)
- `tmrl/config/_internal/_config_interface.py` (0 bytes)
- `tmrl/config/_internal/_config_model.py` (0 bytes)

All three files are empty placeholders from an incomplete refactoring pass. Nothing in the codebase imports them.

**Verification:** `grep -r "_config_agent\|_config_interface\|_config_model" tmrl/` returns nothing after deletion.

---

### Commit 3 ? `refactor: remove deprecated demo-weight fields, PATH_DATA alias`

**Files modified:**
- `tmrl/config/schema/run_bundle.py` ? remove three Pydantic fields marked `"DEPRECATED / unused"`:
  - `demo_sampling_weight`
  - `demo_weight_decay_samples`
  - `demo_weight_decay_slowdown`
- `tmrl/config/constants.py` ? remove the three derived constants (lines 408?410):
  - `DEMO_SAMPLING_WEIGHT`
  - `DEMO_WEIGHT_DECAY_SAMPLES`
  - `DEMO_WEIGHT_DECAY_SLOWDOWN`
- `tmrl/config/__init__.py` ? remove the three constants from `__all__`
- `tmrl/config/paths.py` ? remove `PATH_DATA = TMRL_FOLDER` alias (line 26); no consumers in the codebase
- `tmrl/config/__init__.py` ? remove `PATH_DATA` from `__all__`

**Verification:** `grep -r "DEMO_SAMPLING_WEIGHT\|DEMO_WEIGHT_DECAY\|PATH_DATA" tmrl/` returns nothing after deletion. `make types` passes.

---

### Commit 4 ? `refactor: remove unused re-exports from nn_utils.py`

**File modified:** `tmrl/custom/models/shared/nn_utils.py`

Remove:
- Line 4: `from typing import cast  # noqa: F401 ? kept for potential downstream use`
- Line 9: `import torch.nn.functional as F  # noqa: F401 ? kept for potential downstream use`

Neither symbol is used inside `nn_utils.py` itself. Neither appears in `__all__`. No codebase consumers were found via grep.

**Verification:** `make fmt && make types` passes cleanly.

---

## Explicitly Out of Scope

| Item | Reason excluded |
|---|---|
| `tmrl/memories/`, `tmrl/interfaces/`, `tmrl/models/`, `tmrl/algorithms/`, `tmrl/trackmania/` | Canonical public namespaces per CLAUDE.md ? preserved for external users |
| `USE_RNN = False` constant | Active code gate in `config_objects.py` at 3 check sites; removal requires auditing and deleting the RNN codepath ? separate task |
| `sophy_legacy.py` | Still consumed by `sophy.py` re-export shim; live code |
| `_legacy_action_table` in `_iqn_agent.py` | Active backward-compat buffer migration logic for replay data format |
