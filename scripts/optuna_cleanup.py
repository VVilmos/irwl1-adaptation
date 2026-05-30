import optuna
from optuna.trial import TrialState
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
storage_root = Path("optuna")
if not storage_root.is_absolute():
    storage_root = root_dir / storage_root
storage_root.mkdir(parents=True, exist_ok=True)

study_name = "staticrewind_tune_resnet20"

storage_path = storage_root / f"{study_name}.db"

# Connect to the SQLite file
storage_url = f"sqlite:///{storage_path.as_posix()}"

study = optuna.load_study(
    study_name="resnet20_cifar10",
    storage=storage_url
)

# Find and fix any dead trials interrupted by Slurm
for trial in study.trials:
    if trial.state == TrialState.RUNNING:
        print(f"Fixing interrupted Trial {trial.number} (Setting state to FAIL)")
        study.storage.set_trial_state(trial._trial_id, TrialState.FAIL)

print("SQLite Database cleaned! You can now safely resubmit your main Slurm script.")