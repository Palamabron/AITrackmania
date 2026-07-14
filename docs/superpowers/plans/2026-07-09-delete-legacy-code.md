# Delete Legacy Code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove four categories of dead/deprecated code from the AITrackmania codebase with no public API changes.

**Architecture:** Four independent commits, each targeting one category of dead code. No new files created. Each task is verified with `grep` to confirm nothing imports the deleted symbols, plus `make types && make test` to confirm no regressions.

**Tech Stack:** Python 3.12, uv, ruff, mypy, pytest.

## Global Constraints

- No changes to canonical public namespaces (`tmrl.memories`, `tmrl.interfaces`, `tmrl.models`, `tmrl.algorithms`, `tmrl.trackmania`) ? these are preserved.
- Run `make fmt && make types && make test` after every task and fix any errors before committing.
- 4 pre-existing pytest collection errors and 1 pre-existing test failure are expected ? do not fix them, just confirm they are unchanged.
- `make fmt` fails on macOS via `uv run` due to `vgamepad`; use `.venv/bin/ruff format tmrl/` and `.venv/bin/ruff check tmrl/` directly as a workaround.

---

### Task 1: Remove deprecated `custom_algorithms` package

**Files:**
- Delete: `tmrl/custom/custom_algorithms/__init__.py`
- Delete: `tmrl/custom/custom_algorithms/` (directory)

**Interfaces:**
- Consumes: nothing
- Produces: nothing (deletion only)

- [ ] **Step 1: Confirm no internal consumers**

  ```bash
  grep -r "custom_algorithms" tmrl/
  ```
  Expected: output contains only `tmrl/custom/custom_algorithms/__init__.py` itself. If any other file is listed, stop and investigate before proceeding.

- [ ] **Step 2: Delete the package**

  ```bash
  git rm tmrl/custom/custom_algorithms/__init__.py
  rmdir tmrl/custom/custom_algorithms
  ```

- [ ] **Step 3: Verify clean removal**

  ```bash
  grep -r "custom_algorithms" tmrl/
  ```
  Expected: no output.

- [ ] **Step 4: Run checks**

  ```bash
  .venv/bin/ruff format tmrl/ && .venv/bin/ruff check tmrl/ && .venv/bin/mypy tmrl/ && .venv/bin/pytest tests/ -q
  ```
  Expected: ruff clean, mypy 0 issues in 188 files, same pre-existing test failures as before (4 collection errors + 1 failure, no new ones).

- [ ] **Step 5: Commit**

  ```bash
  git commit -m "refactor: remove deprecated custom_algorithms package"
  ```

---

### Task 2: Delete zero-byte config stubs

**Files:**
- Delete: `tmrl/config/_internal/_config_agent.py`
- Delete: `tmrl/config/_internal/_config_interface.py`
- Delete: `tmrl/config/_internal/_config_model.py`

**Interfaces:**
- Consumes: nothing
- Produces: nothing (deletion only)

- [ ] **Step 1: Confirm files are empty and unimported**

  ```bash
  wc -c tmrl/config/_internal/_config_agent.py tmrl/config/_internal/_config_interface.py tmrl/config/_internal/_config_model.py
  ```
  Expected: all three show 0 bytes.

  ```bash
  grep -r "_config_agent\|_config_interface\|_config_model" tmrl/
  ```
  Expected: no output (nothing imports them).

- [ ] **Step 2: Delete the files**

  ```bash
  git rm tmrl/config/_internal/_config_agent.py \
         tmrl/config/_internal/_config_interface.py \
         tmrl/config/_internal/_config_model.py
  ```

- [ ] **Step 3: Run checks**

  ```bash
  .venv/bin/ruff format tmrl/ && .venv/bin/ruff check tmrl/ && .venv/bin/mypy tmrl/ && .venv/bin/pytest tests/ -q
  ```
  Expected: all clean, same pre-existing failures as baseline.

- [ ] **Step 4: Commit**

  ```bash
  git commit -m "refactor: delete zero-byte config stubs"
  ```

---

### Task 3: Remove deprecated demo-weight fields and `PATH_DATA` alias

**Files:**
- Modify: `tmrl/config/schema/run_bundle.py:175-190`
- Modify: `tmrl/config/constants.py:408-410`
- Modify: `tmrl/config/__init__.py` (import block lines 46-48, `__all__` lines 288-290, import line 231, `__all__` line 358)
- Modify: `tmrl/config/paths.py:26`

**Interfaces:**
- Consumes: nothing from prior tasks
- Produces: nothing (deletion only)

- [ ] **Step 1: Confirm no consumers outside config/**

  ```bash
  grep -r "DEMO_SAMPLING_WEIGHT\|DEMO_WEIGHT_DECAY_SAMPLES\|DEMO_WEIGHT_DECAY_SLOWDOWN\|PATH_DATA" tmrl/ --include="*.py" | grep -v "tmrl/config/"
  ```
  Expected: no output. If anything appears, stop and investigate.

- [ ] **Step 2: Remove three fields from `tmrl/config/schema/run_bundle.py`**

  Find the block starting at line 175 and remove these three field definitions (lines 175?190):
  ```python
      demo_sampling_weight: Annotated[float, Field(ge=0.0)] = Field(
          default=1.0,
          description=(
              "DEPRECATED / unused: not read by any training or memory code path. "
              "Demo exposure is controlled by demo_min/max_batch_fraction instead."
          ),
      )
      demo_weight_decay_samples: int = Field(
          default=0,
          ge=0,
          description=("DEPRECATED / unused: not read by any training or memory code path."),
      )
      demo_weight_decay_slowdown: Annotated[float, Field(ge=0.0)] = Field(
          default=1.0,
          description=("DEPRECATED / unused: not read by any training or memory code path."),
      )
  ```
  Remove all 16 lines. The field before them (`demo_injection_repeat`) and after (`per_alpha`) must remain untouched.

- [ ] **Step 3: Remove three constants from `tmrl/config/constants.py`**

  Find and remove lines 408?410:
  ```python
  DEMO_SAMPLING_WEIGHT = max(0.0, float(PR.demo_sampling_weight))
  DEMO_WEIGHT_DECAY_SAMPLES = max(0, int(PR.demo_weight_decay_samples))
  DEMO_WEIGHT_DECAY_SLOWDOWN = max(0.0, float(PR.demo_weight_decay_slowdown))
  ```

- [ ] **Step 4: Remove `PATH_DATA` alias from `tmrl/config/paths.py`**

  Find and remove line 26:
  ```python
  PATH_DATA = TMRL_FOLDER
  ```

- [ ] **Step 5: Update `tmrl/config/__init__.py` ? remove imports**

  In the `from tmrl.config.constants import (` block (around lines 40?200), remove these three lines:
  ```python
      DEMO_SAMPLING_WEIGHT,
      DEMO_WEIGHT_DECAY_SAMPLES,
      DEMO_WEIGHT_DECAY_SLOWDOWN,
  ```

  In the `from tmrl.config.paths import (` block (around lines 210?240), remove:
  ```python
      PATH_DATA,
  ```

- [ ] **Step 6: Update `tmrl/config/__init__.py` ? remove from `__all__`**

  In the `__all__` list, remove the four string entries:
  ```python
      "DEMO_SAMPLING_WEIGHT",
      "DEMO_WEIGHT_DECAY_SAMPLES",
      "DEMO_WEIGHT_DECAY_SLOWDOWN",
  ```
  and:
  ```python
      "PATH_DATA",
  ```

- [ ] **Step 7: Run checks**

  ```bash
  .venv/bin/ruff format tmrl/ && .venv/bin/ruff check tmrl/ && .venv/bin/mypy tmrl/ && .venv/bin/pytest tests/ -q
  ```
  Expected: all clean, same pre-existing failures as baseline.

- [ ] **Step 8: Verify clean removal**

  ```bash
  grep -r "DEMO_SAMPLING_WEIGHT\|DEMO_WEIGHT_DECAY_SAMPLES\|DEMO_WEIGHT_DECAY_SLOWDOWN\|PATH_DATA" tmrl/
  ```
  Expected: no output.

- [ ] **Step 9: Commit**

  ```bash
  git add tmrl/config/schema/run_bundle.py tmrl/config/constants.py tmrl/config/paths.py tmrl/config/__init__.py
  git commit -m "refactor: remove deprecated demo-weight fields and PATH_DATA alias"
  ```

---

### Task 4: Remove unused re-exports from `nn_utils.py`

**Files:**
- Modify: `tmrl/custom/models/shared/nn_utils.py:4,9`

**Interfaces:**
- Consumes: nothing from prior tasks
- Produces: nothing (deletion only)

- [ ] **Step 1: Confirm no consumers of these re-exports**

  ```bash
  grep -r "from tmrl.custom.models.shared.nn_utils import cast\|from tmrl.custom.models.shared.nn_utils import F\b" tmrl/
  ```
  Expected: no output.

  Also check for star imports that might pull in `cast` or `F` from this module:
  ```bash
  grep -r "from tmrl.custom.models.shared.nn_utils import \*" tmrl/
  ```
  Expected: no output.

- [ ] **Step 2: Remove the two unused import lines from `nn_utils.py`**

  Current lines 4 and 9:
  ```python
  from typing import cast  # noqa: F401 ? kept for potential downstream use
  ```
  and:
  ```python
  import torch.nn.functional as F  # noqa: F401 ? kept for potential downstream use
  ```

  Remove both lines entirely. After removal, the top of the file should look like:
  ```python
  """Constants, basic NN utilities, obs-space helpers, and conv helpers."""

  from math import floor

  import numpy as np
  import torch
  import torch.nn as nn
  from torch.nn import Conv2d

  from tmrl.util import prod
  ```

- [ ] **Step 3: Run checks**

  ```bash
  .venv/bin/ruff format tmrl/ && .venv/bin/ruff check tmrl/ && .venv/bin/mypy tmrl/ && .venv/bin/pytest tests/ -q
  ```
  Expected: all clean, same pre-existing failures as baseline.

- [ ] **Step 4: Commit**

  ```bash
  git add tmrl/custom/models/shared/nn_utils.py
  git commit -m "refactor: remove unused cast and F re-exports from nn_utils"
  ```

---

## Self-Review

**Spec coverage:**
- ? Task 1 ? Commit 1 (custom_algorithms package)
- ? Task 2 ? Commit 2 (zero-byte stubs)
- ? Task 3 ? Commit 3 (deprecated schema fields + PATH_DATA)
- ? Task 4 ? Commit 4 (nn_utils unused re-exports)

**Placeholder scan:** No TBDs, no vague steps ? every step has the exact code or command.

**Type consistency:** No new types or interfaces. All changes are deletions.
