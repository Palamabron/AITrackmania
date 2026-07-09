# Release Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the AITrackmania codebase release-ready by fixing two functional bugs and sweeping for documentation gaps, dead code, and metadata issues.

**Architecture:** Five independent commit-sized tasks executed in severity order. Tasks 1–2 are functional fixes; Tasks 3–5 are cosmetic. Each task ends with `make fmt && make types && make test` passing.

**Tech Stack:** Python 3.12, uv, ruff, mypy, pytest, loguru, Pydantic v2.

## Global Constraints

- All commands run via `uv run` or `make` (see CLAUDE.md).
- Docstring format: Google style (Args / Returns / Raises sections), matching existing style in `tmrl/actor.py` and `tmrl/registry.py`.
- Do not add comments explaining WHAT code does — only add docstrings that describe the public contract.
- Run `make fmt` then `make types` after every task and fix any errors before committing.
- Never skip `make test` before committing.

---

### Task 1: Fix PEP 561 — ship `py.typed` marker in the wheel

**Files:**
- Modify: `pyproject.toml:135-136`

**Interfaces:**
- Produces: `py.typed` marker included in the built wheel, enabling downstream mypy to treat tmrl as typed.

- [ ] **Step 1: Edit `pyproject.toml`**

  In `pyproject.toml`, find the `[tool.setuptools.package-data]` section (currently line 135–136):
  ```toml
  [tool.setuptools.package-data]
  tmrl = ["config/defaults/**/*.yaml"]
  ```
  Change it to:
  ```toml
  [tool.setuptools.package-data]
  tmrl = ["py.typed", "config/defaults/**/*.yaml"]
  ```

- [ ] **Step 2: Verify marker file exists**

  ```bash
  ls tmrl/py.typed
  ```
  Expected: file exists (0 bytes). If missing, create it: `touch tmrl/py.typed`.

- [ ] **Step 3: Run checks**

  ```bash
  make fmt && make types && make test
  ```
  Expected: all pass with no errors.

- [ ] **Step 4: Commit**

  ```bash
  git add pyproject.toml
  git commit -m "fix: ship py.typed marker in wheel (PEP 561)"
  ```

---

### Task 2: Fix plugin-system breakage — replace hardcoded algo whitelist with registry lookup

**Files:**
- Modify: `tmrl/config/config_objects.py:74-75`

**Interfaces:**
- Consumes: `ALGORITHMS` (imported at line 64 from `tmrl.registry`), already fully populated by the side-effect imports on lines 15–19 before the check runs.
- Produces: startup no longer blocks plugin-registered algorithms.

- [ ] **Step 1: Edit `config_objects.py`**

  In `tmrl/config/config_objects.py`, find lines 74–75:
  ```python
  if ALG_NAME not in ("SAC", "REDQSAC", "TQC", "IQN", "SDSAC"):
      raise ValueError(f"Unknown algorithm {ALG_NAME!r}. Supported: SAC, REDQSAC, TQC, IQN, SDSAC.")
  ```
  Replace with:
  ```python
  if ALG_NAME not in ALGORITHMS.keys():
      raise ValueError(
          f"Unknown algorithm {ALG_NAME!r}. "
          f"Registered algorithms: {sorted(ALGORITHMS.keys())}."
      )
  ```

- [ ] **Step 2: Run checks**

  ```bash
  make fmt && make types && make test
  ```
  Expected: all pass. The existing test suite exercises config loading, so this will catch any regression.

- [ ] **Step 3: Commit**

  ```bash
  git add tmrl/config/config_objects.py
  git commit -m "fix: replace hardcoded algo whitelist with registry lookup"
  ```

---

### Task 3: Add Google-format docstrings to public API gaps

**Files:**
- Modify: `tmrl/util.py`
- Modify: `tmrl/wrappers.py`
- Modify: `tmrl/actor.py`
- Modify: `tmrl/envs.py`
- Modify: `tmrl/memory/base.py`
- Modify: `tmrl/networking/trainer.py`

**Interfaces:**
- No interface changes — docstrings only.

- [ ] **Step 1: Document `tmrl/util.py`**

  Add docstrings to these six currently-undocumented functions. Find each by its `def` line and insert the docstring immediately after the `def` line.

  **`pandas_dict` (line 26):**
  ```python
  def pandas_dict(*args, **kwargs) -> pd.Series:
      """Construct a ``pd.Series`` with object dtype from positional or keyword arguments.

      A thin convenience wrapper around ``pd.Series(dict(...), dtype=object)``.

      Args:
          *args: Positional arguments forwarded to ``dict()``.
          **kwargs: Keyword arguments forwarded to ``dict()``.

      Returns:
          pd.Series: Object-typed series keyed by the dict keys.
      """
      return pd.Series(dict(*args, **kwargs), dtype=object)
  ```

  **`shallow_copy` (line 57):**
  ```python
  def shallow_copy[T](obj: T) -> T:
      """Create a shallow copy of any object by copying its ``__dict__``.

      Allocates a new instance of the same type without calling ``__init__``,
      then copies all instance attributes. Does not recurse into attribute values.

      Args:
          obj: The object to copy.

      Returns:
          A new instance of the same type with the same ``__dict__`` entries.
      """
  ```

  **`get_class_or_function` (line 221):**
  ```python
  def get_class_or_function(func):
      """Import and return a class or function by its fully-qualified ``module:name`` string.

      Args:
          func: Dotted module path and name separated by ``:``, e.g. ``"tmrl.actor:ActorModule"``.

      Returns:
          The resolved class or function object.

      Raises:
          ImportError: If the module cannot be imported.
          AttributeError: If the name does not exist in the module.
      """
  ```

  **`partial_from_args` (line 226):**
  ```python
  def partial_from_args(func: str | Callable[..., Any], kwargs: dict[str, str]):
      """Build a ``functools.partial`` from a ``module:name`` string and a flat string dict.

      Resolves nested parameters separated by ``.``, coerces values to their annotated types,
      and handles ``bool`` and sub-``type`` parameters recursively.

      Args:
          func: A ``module:name`` string (resolved via :func:`get_class_or_function`) or a callable.
          kwargs: Flat mapping of parameter names (or ``parent.child`` dotted paths) to string values.

      Returns:
          functools.partial: Partial with resolved and type-coerced keyword arguments.

      Raises:
          AssertionError: If a key in ``kwargs`` is not a valid parameter of ``func``.
      """
  ```

  **`get_output` (line 251):**
  ```python
  def get_output(*args, default="", **kwargs):
      """Run a subprocess and return its stdout, or ``default`` if the process fails.

      Args:
          *args: Positional arguments forwarded to ``subprocess.check_output``.
          default: Value to return when the subprocess exits with a non-zero status.
          **kwargs: Keyword arguments forwarded to ``subprocess.check_output``.

      Returns:
          str: Stripped stdout of the subprocess, or ``default`` on ``CalledProcessError``.
      """
  ```

  **`dump` (line 329):**
  ```python
  def dump(obj, path):
      """Atomically pickle ``obj`` to ``path`` using a temporary file and ``os.replace``.

      If called from the main thread, defers ``SIGINT``/``SIGTERM`` until the write completes
      to prevent partial writes on interrupt.

      Args:
          obj: Any picklable Python object.
          path: Destination file path (``str`` or ``pathlib.Path``).
      """
  ```

  **`load` (line 344):**
  ```python
  def load(path):
      """Unpickle and return an object from ``path``.

      Args:
          path: Source file path (``str`` or ``pathlib.Path``).

      Returns:
          The deserialized Python object.
      """
  ```

  **`save_json` (line 349):**
  ```python
  def save_json(d, path):
      """Serialize ``d`` to a JSON file at ``path`` with UTF-8 encoding and 2-space indentation.

      Args:
          d: A JSON-serializable object (dict, list, str, int, float, bool, or None).
          path: Destination file path (``str`` or ``pathlib.Path``).
      """
  ```

  **`load_json` (line 354):**
  ```python
  def load_json(path):
      """Deserialize and return a JSON object from ``path``.

      Args:
          path: Source file path (``str`` or ``pathlib.Path``).

      Returns:
          The deserialized Python object (dict, list, str, int, float, bool, or None).
      """
  ```

  **`prod` (line 382):**
  ```python
  def prod(iterable):
      """Return the product of all elements in ``iterable``.

      Args:
          iterable: Any iterable of numeric values.

      Returns:
          The product of all elements, starting from 1 for an empty iterable.
      """
  ```

- [ ] **Step 2: Document `tmrl/wrappers.py`**

  **`AffineObservationWrapper` class (line 10) — add class docstring after the class definition line:**
  ```python
  class AffineObservationWrapper(gymnasium.ObservationWrapper):
      """Gymnasium wrapper that applies an affine transform ``(obs + shift) * scale`` to observations.

      Only ``gymnasium.spaces.Box`` observation spaces are supported. The observation
      space bounds are transformed alongside the observations.

      Args:
          env: The environment to wrap.
          shift: Value added to each observation before scaling.
          scale: Value by which the shifted observation is multiplied.
      """
  ```

  **`AffineObservationWrapper.observation` (line 22):**
  ```python
  def observation(self, observation):
      """Apply the affine transform to a single observation.

      Args:
          observation: Raw observation from the wrapped environment.

      Returns:
          Transformed observation: ``(observation + self.shift) * self.scale``.
      """
  ```

  **`float64_to_float32` (line 88):**
  ```python
  def float64_to_float32(x):
      """Cast a ``float64`` numpy array to ``float32``; return other dtypes unchanged.

      Args:
          x: A numpy array.

      Returns:
          numpy.ndarray: ``float32`` array if input dtype is ``float64``, otherwise ``x`` unchanged.
      """
  ```

  **`float_to_float32` (line 101):**
  ```python
  def float_to_float32(x):
      """Wrap a Python float in a single-element ``float32`` numpy array.

      Args:
          x: A Python float or compatible scalar.

      Returns:
          numpy.ndarray: Shape ``(1,)`` array of dtype ``float32``.
      """
  ```

  **`int_to_float32` (line 110):**
  ```python
  def int_to_float32(x):
      """Wrap a Python int in a single-element ``float32`` numpy array.

      Args:
          x: A Python int or compatible scalar.

      Returns:
          numpy.ndarray: Shape ``(1,)`` array of dtype ``float32``.
      """
  ```

- [ ] **Step 3: Document `tmrl/actor.py`**

  **`ActorModule.act_` (line 115):**
  ```python
  def act_(self, obs, test=False):
      """Like :meth:`act`, but may apply pre/post-processing.

      The base implementation simply delegates to :meth:`act`.
      ``TorchActorModule`` overrides this to collate ``obs`` onto the device,
      disable gradients, and clip the returned action to ``[-1, 1]``.

      Args:
          obs: The observation from the environment.
          test (bool): True at test time, False during training.

      Returns:
          numpy.ndarray: The computed action.
      """
  ```

  **`TorchActorModule.save` (line 150):**
  ```python
  def save(self, path):
      """Save model weights as a ``torch.save`` state dict.

      Args:
          path: Destination file path (``str`` or ``pathlib.Path``).
      """
  ```

  **`TorchActorModule.load` (line 153):**
  ```python
  def load(self, path, device):
      """Load state dict from ``path`` and move the model to ``device``.

      Args:
          path: Source file path (``str`` or ``pathlib.Path``).
          device: PyTorch device string or object (e.g. ``"cpu"`` or ``"cuda:0"``).

      Returns:
          TorchActorModule: ``self`` with updated weights on ``device``.
      """
  ```

  **`TorchActorModule.save_to_bytes` (line 158):**
  ```python
  def save_to_bytes(self) -> bytes:
      """Serialize model weights to bytes via an in-memory ``torch.save`` buffer.

      Returns:
          bytes: Serialized state dict, suitable for :meth:`load_from_bytes`.
      """
  ```

  **`TorchActorModule.to` (line 223):**
  ```python
  def to(self, device):  # type: ignore[override]
      """Move the module to ``device`` and update ``self.device``.

      Overrides ``torch.nn.Module.to`` to keep ``self.device`` in sync.

      Args:
          device: PyTorch device string or object.

      Returns:
          TorchActorModule: ``self`` moved to ``device``.
      """
  ```

  **`TorchActorModule.to_device` (line 227):**
  ```python
  def to_device(self, device):
      """Move the module to ``device`` by delegating to :meth:`to`.

      Args:
          device: PyTorch device string or object.

      Returns:
          TorchActorModule: ``self`` moved to ``device``.
      """
  ```

- [ ] **Step 4: Document `tmrl/envs.py`**

  The `__init__` already has a docstring. Add a class-level docstring to `GenericGymEnv` (line 12), immediately after `class GenericGymEnv(gymnasium.Wrapper):`:
  ```python
  class GenericGymEnv(gymnasium.Wrapper):
      """Lightweight Gymnasium wrapper for use with the TMRL framework.

      Optionally applies an affine observation rescaling and/or a float32 cast.
      Use this when integrating arbitrary Gymnasium environments with TMRL.
      """
  ```

- [ ] **Step 5: Document `tmrl/memory/base.py`**

  **`Memory.sample` (line 144):**
  ```python
  def sample(self):
      """Sample a batch of transitions and collate them onto ``self.device``.

      Calls :meth:`sample_indices` to draw indices, retrieves each transition via
      ``__getitem__``, then colates the batch with :meth:`collate`.

      Returns:
          Tuple of tensors as returned by :meth:`collate`.

      Raises:
          RuntimeError: If the buffer does not have enough data to draw a valid batch
              (e.g. fewer transitions than ``n_step_return``).
      """
  ```

  **`Memory.append` (line 155):**
  ```python
  def append(self, buffer):
      """Append a :class:`~tmrl.networking.Buffer` to the replay memory.

      Copies episode statistics from ``buffer`` into ``self.stat_*`` fields
      and delegates storage to :meth:`append_buffer`. No-ops if ``buffer`` is empty.

      Args:
          buffer (tmrl.networking.Buffer): Buffer of transitions received from a worker.
      """
  ```

  **`Memory.sample_indices` (line 205):**
  ```python
  def sample_indices(self):
      """Return a batch of randomly sampled transition indices.

      When ``n_step_return > 1``, samples from ``[0, len - n_step_return]`` so that
      ``n`` consecutive transitions are always available for each index.

      Returns:
          numpy.ndarray | tuple: Array of ``int64`` indices of length ``batch_size``,
          or an empty tuple if the buffer has insufficient data.
      """
  ```

- [ ] **Step 6: Document `tmrl/networking/trainer.py`**

  **`TrainerInterface.__init__` (line 285) — currently has no docstring:**
  ```python
  def __init__(
      self,
      server_ip=None,
      ...
  ):
      """Connect to the relay server as the trainer endpoint.

      Args:
          server_ip (str | None): IP address of the relay server. Defaults to ``"127.0.0.1"``.
          server_port (int): Public port of the relay server.
          password (str): Shared password for the relay server.
          local_com_port (int): Local port for the tlspyo endpoint.
          header_size (int): Byte size of the tlspyo message header.
          max_buf_len (int): Maximum buffer length for incoming messages.
          security (str): Security mode (``"TLS"`` or ``"TCP"``).
          keys_dir (str | Path): Directory containing TLS credentials.
          hostname (str): Server hostname for TLS verification.
          model_path (str | Path): Path used for temporary model weight files.
      """
  ```

  **`TrainerInterface.broadcast_model` (line 321) — replace terse docstring:**
  ```python
  def broadcast_model(self, model: ActorModule):
      """Serialize and broadcast model weights to all connected workers.

      Prefers :meth:`~tmrl.actor.ActorModule.save_to_bytes` when available (in-memory,
      no disk I/O). Falls back to saving to ``self.model_path`` and reading it back.

      Args:
          model (ActorModule): The actor whose weights to broadcast.
      """
  ```

  **`TrainerInterface.retrieve_buffer` (line 334) — replace terse docstring:**
  ```python
  def retrieve_buffer(self):
      """Receive all pending buffers from the server and merge them into one.

      Returns:
          tmrl.networking.Buffer: Merged buffer containing all received transitions.
              Returns an empty ``Buffer`` if no data is available.
      """
  ```

- [ ] **Step 7: Run checks and commit**

  ```bash
  make fmt && make types && make test
  ```
  Expected: all pass.

  ```bash
  git add tmrl/util.py tmrl/wrappers.py tmrl/actor.py tmrl/envs.py tmrl/memory/base.py tmrl/networking/trainer.py
  git commit -m "docs: add Google-format docstrings to public API gaps"
  ```

---

### Task 4: Remove dead code and clean up inline comments

**Files:**
- Modify: `tmrl/envs.py`
- Modify: `tmrl/actor.py`
- Modify: `tmrl/custom/utils/nn_distributions.py`
- Modify: `tmrl/custom/tm/utils/control/gamepad.py`
- Modify: `tmrl/tools/diagnostics/check_environment.py`
- Modify: `tmrl/tuto/tuto_minimal_drone.py`

**Interfaces:**
- No behavior changes. All modifications are comment/dead-code removals or print→logger conversions.

- [ ] **Step 1: Clean `tmrl/envs.py`**

  Remove the commented-out lines (32–33) and the empty `__main__` guard (37–38).

  Current state (lines 25–38):
  ```python
      super().__init__(env)


  if __name__ == "__main__":
      pass
  ```
  And lines 32–33 inside `__init__`:
  ```python
      # assert isinstance(env.action_space, gymnasium.spaces.Box), f"{env.action_space}"
      # env = NormalizeActionWrapper(env)
      super().__init__(env)
  ```

  After edit, the two comment lines and the `if __name__` block are removed. The final file ends at `super().__init__(env)`.

- [ ] **Step 2: Clean `tmrl/actor.py`**

  Remove line 147: `# super().__init__()  # torch.nn.Module`

  Find:
  ```python
          super().__init__(observation_space, action_space)  # ActorModule
          # super().__init__()  # torch.nn.Module
          self.device = device
  ```
  Replace with:
  ```python
          super().__init__(observation_space, action_space)  # ActorModule
          self.device = device
  ```

- [ ] **Step 3: Clean `tmrl/custom/utils/nn_distributions.py`**

  Remove the commented-out line 85: `# a = TanhTransformedDist(Independent(Normal(m, std), 1))`

  Find:
  ```python
          # a = TanhTransformedDist(Independent(Normal(m, std), 1))
          a = Independent(TanhNormal(mean, std), 1)
  ```
  Replace with:
  ```python
          a = Independent(TanhNormal(mean, std), 1)
  ```

- [ ] **Step 4: Clean `tmrl/custom/tm/utils/control/gamepad.py`**

  Remove the commented-out line 24: `# mapped_value = 0.5 * control[0] + 0.5  # x0 = 1/2`

  Find:
  ```python
          if control[1] > 0.75:  # break
              # mapped_value = 0.5 * control[0] + 0.5  # x0 = 1/2
              gamepad.left_trigger_float(value_float=control[1])
  ```
  Replace with:
  ```python
          if control[1] > 0.75:  # break
              gamepad.left_trigger_float(value_float=control[1])
  ```

- [ ] **Step 5: Fix `tmrl/tools/diagnostics/check_environment.py`**

  Convert `print` calls on lines 30–31 to `logger.debug`. The function already imports `logger` at the top of the file.

  Find:
  ```python
          if d or t:
              print("d: ", d)
              print("t: ", t)
              _o, _ = env.reset()
  ```
  Replace with:
  ```python
          if d or t:
              logger.debug("d: {}", d)
              logger.debug("t: {}", t)
              _o, _ = env.reset()
  ```

- [ ] **Step 6: Clean `tmrl/tuto/tuto_minimal_drone.py`**

  Remove the commented-out line 145: `# env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", ...)`

  Find:
  ```python
  # Dummy environment OR (observation space, action space) tuple:
  # env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_rtgym_config})
  env_cls = (obs_space, act_space)
  ```
  Replace with:
  ```python
  # Dummy environment OR (observation space, action space) tuple:
  env_cls = (obs_space, act_space)
  ```

- [ ] **Step 7: Run checks and commit**

  ```bash
  make fmt && make types && make test
  ```
  Expected: all pass.

  ```bash
  git add tmrl/envs.py tmrl/actor.py tmrl/custom/utils/nn_distributions.py \
      tmrl/custom/tm/utils/control/gamepad.py \
      tmrl/tools/diagnostics/check_environment.py \
      tmrl/tuto/tuto_minimal_drone.py
  git commit -m "refactor: remove dead code and clean up inline comments"
  ```

---

### Task 5: Fix `pyproject.toml` metadata — move pyinstrument to dev, delete stale readthedocs config

**Files:**
- Modify: `pyproject.toml`
- Delete: `.readthedocs.yaml`

**Interfaces:**
- No code changes. Users who enable `python_profiling=True` need to install the `dev` extra or install `pyinstrument` manually.

- [ ] **Step 1: Move `pyinstrument` in `pyproject.toml`**

  In `[project.dependencies]` (line 53), remove:
  ```toml
      "pyinstrument",
  ```

  In `[project.optional-dependencies] dev` (lines 74–79), add `pyinstrument`:
  ```toml
  [project.optional-dependencies]
  dev = [
      "mypy>=1.8",
      "pyinstrument",
      "ruff>=0.4",
      "pytest>=7.0",
      "types-PyYAML",
  ]
  ```

  Also update the `[dependency-groups] dev` section (lines 84–90) identically:
  ```toml
  [dependency-groups]
  dev = [
      "mypy>=1.8",
      "pyinstrument",
      "ruff>=0.4",
      "pytest>=7.0",
      "types-PyYAML",
  ]
  ```

- [ ] **Step 2: Delete stale ReadTheDocs v1 config**

  ```bash
  git rm .readthedocs.yaml
  ```

- [ ] **Step 3: Run checks and commit**

  ```bash
  make fmt && make types && make test
  ```
  Expected: all pass.

  ```bash
  git add pyproject.toml
  git commit -m "chore: move pyinstrument to dev deps, remove stale readthedocs v1 config"
  ```

---

## Self-Review

**Spec coverage check:**
- ✅ Task 1 → Commit 1 (py.typed PEP 561)
- ✅ Task 2 → Commit 2 (algo whitelist → registry lookup)
- ✅ Task 3 → Commit 3 (docstrings: util, wrappers, actor, envs, memory/base, trainer)
- ✅ Task 4 → Commit 4 (dead code: envs, actor, nn_distributions, gamepad, check_environment, tuto)
- ✅ Task 5 → Commit 5 (pyinstrument → dev, delete .readthedocs.yaml)

**Placeholder scan:** No TBDs, TODOs, or vague requirements found.

**Type consistency:** No new types or interfaces introduced — all changes are to existing signatures.
