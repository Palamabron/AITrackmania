"""
Pobiera ostatnie 50 rund z runu WandB i wypisuje je.
Uruchom w środowisku z zainstalowanym wandb (np. tam gdzie trenujesz):
  python scripts/wandb_last_50_rounds.py

Jeśli w projekcie jest folder wandb/ (artefakty), uruchom z innego katalogu
lub: python -c "exec(open('scripts/wandb_last_50_rounds.py').read())"
z PYTHONPATH bez katalogu projektu (żeby załadować bibliotekę wandb).
"""
import sys
import os

# Jeśli skrypt jest w AITrackmania/scripts/, usuń repo z path przed importem wandb,
# żeby załadować bibliotekę wandb z site-packages, nie lokalny folder wandb/
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _repo_root in sys.path:
    sys.path.remove(_repo_root)
# Dodaj z powrotem po imporcie wandb, żeby ewentualne inne importy działały
try:
    import wandb
except Exception as e:
    # Przywróć path i spróbuj normalnie (np. gdy uruchamiasz z innego katalogu)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    import wandb

if not hasattr(wandb, "Api"):
    print("Uwaga: załadowano lokalny pakiet wandb zamiast biblioteki. Uruchom skrypt z innego katalogu, np.:")
    print("  cd %TEMP% && python H:\\Studia\\...\\AITrackmania\\scripts\\wandb_last_50_rounds.py")
    sys.exit(1)

api = wandb.Api()
run = api.run("tmrl/tmrl/SophyResidual_runv23_RUN_Hv2 TRAINER")
h = run.history()

print("Kolumny:", list(h.columns))
print()
print("=== Ostatnie 50 rund ===")
last50 = h.tail(50)
print(last50.to_string())
print()
print("=== Podsumowanie (ostatnie 50) ===")
for col in ["return_train", "episode_length_train", "debug/demo_fraction_in_batch", "debug/q_a1", "losses/loss_actor"]:
    if col in last50.columns:
        s = last50[col].dropna()
        if len(s) > 0:
            print(f"  {col}: min={s.min():.4g}, max={s.max():.4g}, mean={s.mean():.4g}, last={s.iloc[-1]:.4g}")
