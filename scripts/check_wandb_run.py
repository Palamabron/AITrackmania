"""Check if a wandb run is currently training (running). Uses WANDB_API_KEY from .env."""

import os
import sys

# Load .env from repo root
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
_env = os.path.join(_repo_root, ".env")
if os.path.isfile(_env):
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k == "WANDB_API_KEY":
                    os.environ["WANDB_API_KEY"] = v
                    break

# Avoid loading local wandb package
if _repo_root in sys.path:
    sys.path.remove(_repo_root)
import wandb

api = wandb.Api()

# Paths to try (user path: entity/project/run_id; "runs/" might be wrong)
paths_to_try = [
    "tmrl/tmrl/SophyResidual_runv23_RUN_L TRAINER",
    "tmrl/tmrl/runs/SophyResidual_runv23_RUN_L TRAINER",
]

run = None
for path in paths_to_try:
    try:
        run = api.run(path)
        break
    except Exception as e:
        print(f"  {path}: {e}")
        continue

if run is None:
    print("Could not find run with given paths.")
    sys.exit(1)

state = run.state  # "running", "finished", "crashed", etc.
print(f"Run: {run.path}")
print(f"State: {state}")
print(f"Training (run is live): {state == 'running'}")

# Czy model się uczył? (pomijamy błąd na końcu – patrzymy na postęp metryk)
try:
    h = run.history()
    n = len(h)
    print(f"\n--- Czy model się uczył? (liczba kroków z logami: {n}) ---")
    if n == 0:
        print("Brak zlogowanych kroków → nie ma danych, by ocenić uczenie.")
    else:
        # Nagroda / return
        for col in [
            "return_train",
            "episode_length_train",
            "losses/loss_actor",
            "losses/loss_critic",
        ]:
            if col not in h.columns:
                continue
            s = h[col].dropna()
            if len(s) < 2:
                continue
            first_vals, last_vals = s.head(max(1, n // 10)), s.tail(max(1, n // 10))
            first_mean, last_mean = first_vals.mean(), last_vals.mean()
            trend = "wzrost" if last_mean > first_mean else "spadek"
            print(f"  {col}: na początku ~{first_mean:.4g}, na końcu ~{last_mean:.4g} ({trend})")
        # Krótkie podsumowanie
        if "return_train" in h.columns:
            r = h["return_train"].dropna()
            if len(r) >= 2:
                improvement = (
                    r.tail(max(1, len(r) // 10)).mean() - r.head(max(1, len(r) // 10)).mean()
                )
                print(f"\n  return_train: zmiana (koniec - początek) ≈ {improvement:+.4g}")
                if improvement > 0:
                    print("  → Tak, model się uczył (nagroda rosła).")
                else:
                    print(
                        "  → Nagroda nie rosła w tym runie (może za mało kroków albo trudna konfiguracja)."
                    )
except Exception as e:
    print(f"\nNie udało się pobrać historii (pominąć ten błąd): {e}")

if run.summary:
    print("\nSummary (last metrics):")
    for k, v in list(run.summary.items())[:15]:
        if not k.startswith("_"):
            print(f"  {k}: {v}")
