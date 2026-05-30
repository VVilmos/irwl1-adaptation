from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import optuna
import torch
import os

import irwl1.config as config
from irwl1.data import fetch_cifar10
from irwl1.model import ResNet20
from irwl1.regularization import L1_penalty_init
from irwl1.utils import (
    calculate_real_sparsity,
    global_pruning,
    init_mask,
    rewind_model_to_checkpoint,
    train,
    train_regularized,
    validate,
    zero_pruned_optimizer_state,
)


MAX_SPARSITY = 90.0
DEFAULT_STUDY_NAME = "resnet20_static_pruning"


def calculate_auc_sac(sparsity_list, accuracy_list):
    """Calculate the normalized area under the sparsity-accuracy curve."""
    sparsities = np.asarray(sparsity_list, dtype=float)
    accuracies = np.asarray(accuracy_list, dtype=float)

    if len(sparsities) != len(accuracies):
        raise ValueError(
            f"List length mismatch: Sparsity ({len(sparsities)}) vs Accuracy ({len(accuracies)})"
        )
    if len(sparsities) < 2:
        return 0.0

    sort_indices = np.argsort(sparsities)
    sorted_sparsities = sparsities[sort_indices]
    sorted_accuracies = accuracies[sort_indices]

    span = float(sorted_sparsities[-1] - sorted_sparsities[0])
    if span <= 0.0:
        return 0.0

    auc = np.trapz(y=sorted_accuracies, x=sorted_sparsities)
    return float(auc / span)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _configure_training() -> None:
    config.WEIGHT_PRUNING_THRESHOLD = 1e-5
    config.KERNEL_PRUNING_THRESHOLD = 3e-5
    config.MODE = "weight-wise"
    config.EPSILON_START = 1.0
    config.EPSILON_END = 1e-6
    config.EPSILON_DECAY_STEPS = 30
    config.IS_EPSILON_DECAY = False
    config.LAMBDA_REG_START = 1e-14
    config.LAMBDA_REG_GROWTH_STEPS = 20
    config.IS_LAMBDA_RISE = False
    config.UPDATE_INTERVAL = 1.0
    config.REG_TYPE = "WL1"
    config.MODEL = "ResNet20"
    config.WANDB_MODE = "disabled"
    config.WEIGHT_DECAY = 1e-4
    config.LEARNING_RATE = 0.001
    config.REWIND = True
    config.REWIND_EPOCH = 4
    config.MAX_EPOCHS = 100
    config.PERSISTENT = False


def _build_model(device: torch.device) -> torch.nn.Module:
    model = ResNet20().to(device)
    L1_penalty_init(model)
    init_mask(model)
    return model


def _build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )


def _apply_trial_parameters(trial: optuna.Trial) -> dict[str, float]:
    epsilon_end = trial.suggest_float("epsilon_end", 1e-8, 1e-2, log=True)
    lambda_reg_end = trial.suggest_float("lambda_reg_end", 1e-10, 1e-1, log=True)
    update_interval = trial.suggest_float("update_interval", 0.5, 3.0)

    config.EPSILON_END = epsilon_end
    config.LAMBDA_REG_END = lambda_reg_end
    config.UPDATE_INTERVAL = update_interval

    return {
        "epsilon_end": epsilon_end,
        "lambda_reg_end": lambda_reg_end,
        "update_interval": update_interval,
    }


def _trial_directory(root_dir: Path, study_name: str, trial_number: int) -> Path:
    trial_dir = root_dir / "optuna" / study_name / f"trial_{trial_number:05d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    return trial_dir


def _run_single_trial(
    trial: optuna.Trial,
    train_loader,
    val_loader,
    device: torch.device,
    root_dir: Path,
    study_name: str,
    rewind: bool,
    max_cycles: int,
) -> float:
    params = _apply_trial_parameters(trial)
    trial_dir = _trial_directory(root_dir, study_name, trial.number)
    rewind_checkpoint_path = trial_dir / "rewind_checkpoint.pth"

    model = _build_model(device)
    optimizer = _build_optimizer(model)

    model, _, _ = train(
        model,
        train_loader,
        val_loader,
        optimizer=optimizer,
        is_new_run=False,
        run=None,
        run_name=f"optuna_trial_{trial.number:05d}",
        weight_decay=True,
        rewind=rewind,
        rewind_checkpoint_path=rewind_checkpoint_path,
        rewind_epoch=config.REWIND_EPOCH,
    )

    sparsity_points = [calculate_real_sparsity(model)]
    _, accuracy = validate(model, val_loader)
    accuracy_points = [accuracy]

    pruning_cycles = 0
    current_real_sparsity = sparsity_points[-1]

    while current_real_sparsity < MAX_SPARSITY:
        model, _, _ = train_regularized(
            model,
            train_loader,
            val_loader,
            optimizer=optimizer,
            is_new_run=False,
            run=None,
            run_name=f"optuna_trial_{trial.number:05d}",
            weight_decay=False,
        )

        global_pruning(model, masking=True)

        if rewind:
            if rewind_checkpoint_path.exists():
                model = rewind_model_to_checkpoint(model, rewind_checkpoint_path)
                optimizer = _build_optimizer(model)
            else:
                zero_pruned_optimizer_state(model, optimizer)
        else:
            zero_pruned_optimizer_state(model, optimizer)

        model, _, _ = train(
            model,
            train_loader,
            val_loader,
            optimizer=optimizer,
            is_new_run=False,
            run=None,
            run_name=f"optuna_trial_{trial.number:05d}",
            weight_decay=True,
            rewind=False,
            rewind_checkpoint_path=rewind_checkpoint_path,
            rewind_epoch=config.REWIND_EPOCH,
        )

        _, accuracy = validate(model, val_loader)
        current_real_sparsity = calculate_real_sparsity(model)
        sparsity_points.append(current_real_sparsity)
        accuracy_points.append(accuracy)
        pruning_cycles += 1

        trial.set_user_attr("current_sparsity", current_real_sparsity)
        trial.set_user_attr("current_accuracy", accuracy)

        if pruning_cycles >= max_cycles and current_real_sparsity < MAX_SPARSITY:
            raise RuntimeError(
                f"Trial {trial.number} did not reach {MAX_SPARSITY:.2f}% sparsity after {max_cycles} pruning cycles"
            )

    objective = calculate_auc_sac(sparsity_points, accuracy_points)
    trial.set_user_attr("sparsity_points", sparsity_points)
    trial.set_user_attr("accuracy_points", accuracy_points)
    trial.set_user_attr("objective_auc_sac", objective)
    trial.set_user_attr("epsilon_end", params["epsilon_end"])
    trial.set_user_attr("lambda_reg_end", params["lambda_reg_end"])
    trial.set_user_attr("update_interval", params["update_interval"])

    return objective


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuner for iterative pruning")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials to run")
    parser.add_argument("--n-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), help="Number of Optuna trials to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--study-name", type=str, default=DEFAULT_STUDY_NAME, help="Optuna study name")
    parser.add_argument("--storage-dir", type=str, default="optuna", help="Directory for Optuna storage")
    parser.add_argument("--max-sparsity", type=float, default=MAX_SPARSITY, help="Target sparsity for each trial")
    parser.add_argument("--max-cycles", type=int, default=30, help="Safety cap on pruning cycles per trial")
    parser.add_argument("--rewind", dest="rewind", action="store_true", help="Enable rewinding after pruning")
    parser.add_argument("--no-rewind", dest="rewind", action="store_false", help="Disable rewinding after pruning")
    parser.set_defaults(rewind=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)

    root_dir = Path(__file__).resolve().parents[1]
    storage_root = Path(args.storage_dir)
    if not storage_root.is_absolute():
        storage_root = root_dir / storage_root
    storage_root.mkdir(parents=True, exist_ok=True)

    _configure_training()
    global MAX_SPARSITY
    MAX_SPARSITY = args.max_sparsity

    train_loader, val_loader, _ = fetch_cifar10()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    storage_path = storage_root / f"{args.study_name}.db"
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=f"sqlite:///{storage_path.as_posix()}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    def objective(trial: optuna.Trial) -> float:
        return _run_single_trial(
            trial=trial,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            root_dir=root_dir,
            study_name=args.study_name,
            rewind=args.rewind,
            max_cycles=args.max_cycles,
        )

    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_workers)


if __name__ == "__main__":
    main()