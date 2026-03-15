import os

import wandb

os.environ["WANDB_API_KEY"] = (
    "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"
)

api = wandb.Api()

run_paths = [
    "/tmrl/tmrl/runs/GnnEffNet_TQC_gSDE_run_Iv5 WORKER",
    "/tmrl/tmrl/runs/GnnEffNet_TQC_gSDE_run_Iv5 TRAINER",
    "tmrl/tmrl/GnnEffNet_TQC_gSDE_run_Iv5 WORKER",
    "tmrl/tmrl/GnnEffNet_TQC_gSDE_run_Iv5 TRAINER",
]

for rp in run_paths:
    try:
        print(f"\nTrying to fetch: {rp}")
        run = api.run(rp)
        print(f"SUCCESS: Found run at {rp}")
        hist = run.history(samples=2000)
        print(f"Columns: {list(hist.columns)}")

        for col in hist.columns:
            if col.startswith("_"):
                continue
            series = hist[col].dropna()
            if len(series) == 0:
                continue

            step = max(1, len(series) // 10)
            progression = series.iloc[::step].values[:10]

            start_val = series.iloc[:5].mean() if len(series) >= 5 else series.iloc[0]
            end_val = series.iloc[-5:].mean() if len(series) >= 5 else series.iloc[-1]
            min_val = series.min()
            max_val = series.max()

            print(f"  Metric: {col}")
            print(f"    Min: {min_val:.4f}, Max: {max_val:.4f}")
            print(f"    Start: {start_val:.4f}, End: {end_val:.4f}")
            prog_str = "[" + ", ".join([f"{x:.4f}" for x in progression]) + "]"
            print(f"    Progression: {prog_str}")
    except Exception as e:
        print(f"Failed. Error: {e}")
