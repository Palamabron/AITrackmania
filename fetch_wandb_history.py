import json

import wandb

api = wandb.Api()

run = None
run_path_attempts = [
    "tmrl/tmrl/GnnEffNet_SophyResidual_TQC_run_Cv4 TRAINER",
]

# Attempt 1: try direct paths
for path in run_path_attempts:
    try:
        print(f"Trying direct path: {path}")
        run = api.run(path)
        print("  -> SUCCESS")
        break
    except Exception as e:
        print(f"  -> Failed: {e}")

# Attempt 2: search by name in project tmrl/tmrl
if run is None:
    print("\nSearching for run by name in project tmrl/tmrl ...")
    try:
        runs = api.runs(
            "tmrl/tmrl", filters={"display_name": {"$regex": "GnnEffNet_SophyResidual_TQC_run_C"}}
        )
        for r in runs:
            print(f"  Found: {r.name}  (id={r.id}, path={r.path})")
        if runs:
            run = runs[0]
            print(f"  -> Using first match: {run.name}")
    except Exception as e:
        print(f"  -> Search failed: {e}")

# Attempt 3: broader search
if run is None:
    print("\nBroadening search (listing recent runs in tmrl/tmrl) ...")
    try:
        runs = api.runs("tmrl/tmrl", per_page=50)
        for r in runs:
            print(f"  {r.name}  (id={r.id})")
            if "SophyResidual" in r.name or "run_C" in r.name:
                run = r
                print("    ^ matched!")
                break
    except Exception as e:
        print(f"  -> Listing failed: {e}")

if run is None:
    print("\nERROR: Could not find the run. Exiting.")
    exit(1)

print(f"\n{'=' * 80}")
print(f"RUN: {run.name}  (id={run.id})")
print(f"URL: {run.url}")
print(f"State: {run.state}")
print(f"{'=' * 80}")

# Config
print("\n--- run.config (hyperparameters) ---")
print(json.dumps(dict(run.config), indent=2, default=str))

# Summary
print("\n--- run.summary (final metrics) ---")
summary_dict = {}
for k, v in run.summary.items():
    try:
        json.dumps(v)
        summary_dict[k] = v
    except (TypeError, ValueError):
        summary_dict[k] = str(v)
print(json.dumps(summary_dict, indent=2, default=str))

# History
print("\n--- Fetching history (samples=10000) ---")
hist = run.history(samples=10000)
print(f"Shape: {hist.shape}")
print(f"Columns ({len(hist.columns)}): {list(hist.columns)}")

print("\nFirst 5 rows:")
print(hist.head().to_string())

print("\nLast 5 rows:")
print(hist.tail().to_string())

# Save
out_path = "/mnt/h/Studia/inzynierskie/inzynierkav2/AITrackmania/wandb_history.csv"
hist.to_csv(out_path, index=False)
print(f"\nSaved {hist.shape[0]} rows to {out_path}")
