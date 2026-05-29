from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from statistics import stdev

import pandas as pd
import torch

from irwl1.data import fetch_cifar10
from irwl1.eval_utils import configure_evaluation, load_model_from_checkpoint, compute_jacobian_norm, compute_layer_condition_numbers

from irwl1.utils import calculate_real_sparsity, test

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_ROOT = PROJECT_ROOT / "models"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "resnet20_pruning_config_summary.csv"
ALLOWED_CONFIGS = ("smooth_rewind", "smooth_no_rewind", "static_rewind", "static_no_rewind")
TARGET_FILENAME_CANDIDATES = ("epochs_by_iteration.csv", "epochs_by_iterations.csv")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Summarize pruning configurations across runs")
	parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT, help="Root directory containing config folders")
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to the summary CSV to write")
	parser.add_argument("--target-sparsity", type=float, default=90.0, help="Target sparsity threshold used to stop counting iterations")
	return parser.parse_args()


def _resolve_existing_path(path: Path) -> Path:
	if path.is_absolute():
		return path
	return (PROJECT_ROOT / path).resolve()


def _numeric_sort_key(path: Path) -> tuple[int, int | str]:
	if path.name.isdigit():
		return (0, int(path.name))
	return (1, path.name)


def _discover_config_dirs(models_root: Path) -> list[Path]:
	config_dirs = [models_root / config_name for config_name in ALLOWED_CONFIGS if (models_root / config_name).is_dir()]
	missing = [config_name for config_name in ALLOWED_CONFIGS if not (models_root / config_name).is_dir()]
	if missing:
		raise FileNotFoundError(f"Missing configuration directories under {models_root}: {', '.join(missing)}")
	return config_dirs


def _discover_run_dirs(config_dir: Path) -> list[Path]:
	run_dirs = sorted((path for path in config_dir.iterdir() if path.is_dir() and path.name.isdigit()), key=_numeric_sort_key)
	if len(run_dirs) < 5:
		raise FileNotFoundError(f"Expected at least 5 run directories in {config_dir}, found {len(run_dirs)}")
	selected_run_dirs = run_dirs[:5]
	extra_run_dirs = run_dirs[5:]
	if extra_run_dirs:
		print(
			f"[WARN] Ignoring extra run directories in {config_dir}: "
			+ ", ".join(path.name for path in extra_run_dirs)
		)
	return selected_run_dirs


def _load_total_epochs(run_dir: Path) -> int:
	csv_path = run_dir / "total_epochs.csv"
	if not csv_path.is_file():
		raise FileNotFoundError(f"Missing {csv_path}")
	frame = pd.read_csv(csv_path)
	if frame.empty:
		raise ValueError(f"Empty total epochs file: {csv_path}")
	return int(frame.iloc[0, 0])


def _load_iteration_frame(run_dir: Path) -> pd.DataFrame:
	for filename in TARGET_FILENAME_CANDIDATES:
		csv_path = run_dir / filename
		if csv_path.is_file():
			frame = pd.read_csv(csv_path)
			if frame.shape[1] < 4:
				raise ValueError(f"Expected at least four columns in {csv_path}, found {frame.shape[1]}")
			if len(frame) < 2:
				raise ValueError(f"Expected at least two rows in {csv_path}, found {len(frame)}")
			return frame
	raise FileNotFoundError(f"Missing epochs-by-iteration CSV in {run_dir}")


def _iteration_metrics_from_frame(frame: pd.DataFrame, target_sparsity: float) -> dict[str, float]:
	train_epochs = frame.iloc[:, 1].astype(float).tolist()
	reg_epochs = frame.iloc[:, 2].astype(float).tolist()
	sparsities = frame.iloc[:, 3].astype(float).tolist()

	target_indices = [index for index, sparsity in enumerate(sparsities[1:], start=1) if sparsity >= target_sparsity]
	if not target_indices:
		raise ValueError(f"Target sparsity {target_sparsity} was never reached")

	target_row_index = target_indices[0]
	iteration_count = float(target_row_index)
	avg_reg_length = float(sum(reg_epochs[:target_row_index]) / target_row_index)
	avg_train_length = float(sum(train_epochs[1 : target_row_index + 1]) / target_row_index)

	return {
		"iterations_to_target": iteration_count,
		"avg_training_length": avg_train_length,
		"avg_regularization_length": avg_reg_length,
	}


def _extract_checkpoint_sparsity(checkpoint_path: Path) -> float:
	match = re.search(r"checkpoint_spar([0-9]+(?:\.[0-9]+)?)\.pth$", checkpoint_path.name)
	if match is None:
		return float("-inf")
	return float(match.group(1))


def _discover_final_checkpoint(run_dir: Path) -> Path:
	checkpoints = list(run_dir.glob("checkpoint_*.pth"))
	if not checkpoints:
		raise FileNotFoundError(f"No checkpoints found in {run_dir}")
	return max(
		checkpoints,
		key=lambda checkpoint_path: (
			_extract_checkpoint_sparsity(checkpoint_path),
			checkpoint_path.stat().st_mtime_ns,
		),
	)


def _sample_std(values: list[float]) -> float:
	if len(values) < 2:
		return float("nan")
	return float(stdev(values))


def _evaluate_run(run_dir: Path, device: torch.device, test_loader, target_sparsity: float) -> dict[str, float]:
	total_epochs = float(_load_total_epochs(run_dir))
	frame = _load_iteration_frame(run_dir)
	iteration_metrics = _iteration_metrics_from_frame(frame, target_sparsity)
	checkpoint_path = _discover_final_checkpoint(run_dir)
	model, _ = load_model_from_checkpoint(checkpoint_path, device)
	_, test_accuracy = test(model, test_loader)
	real_sparsity = calculate_real_sparsity(model)

	return {
		"total_epochs": total_epochs,
		"iterations_to_target": iteration_metrics["iterations_to_target"],
		"avg_training_length": iteration_metrics["avg_training_length"],
		"avg_regularization_length": iteration_metrics["avg_regularization_length"],
		"sparsity_reached": float(real_sparsity),
		"test_accuracy": float(test_accuracy),
	}


def _summarize_config(config_name: str, config_dir: Path, device: torch.device, test_loader, target_sparsity: float) -> dict[str, float | str]:
	run_dirs = _discover_run_dirs(config_dir)
	run_metrics = [_evaluate_run(run_dir, device, test_loader, target_sparsity) for run_dir in run_dirs]

	total_epochs = [metrics["total_epochs"] for metrics in run_metrics]
	iterations_to_target = [metrics["iterations_to_target"] for metrics in run_metrics]
	avg_training_length = [metrics["avg_training_length"] for metrics in run_metrics]
	avg_regularization_length = [metrics["avg_regularization_length"] for metrics in run_metrics]
	sparsity_reached = [metrics["sparsity_reached"] for metrics in run_metrics]
	test_accuracy = [metrics["test_accuracy"] for metrics in run_metrics]

	return {
		"configuration": config_name,
		"num_runs": len(run_metrics),
		"epochs_to_target_mean": float(sum(total_epochs) / len(total_epochs)),
		"epochs_to_target_std": _sample_std(total_epochs),
		"iterations_to_target_mean": float(sum(iterations_to_target) / len(iterations_to_target)),
		"iterations_to_target_std": _sample_std(iterations_to_target),
		"avg_training_length_mean": float(sum(avg_training_length) / len(avg_training_length)),
		"avg_training_length_std": _sample_std(avg_training_length),
		"avg_regularization_length_mean": float(sum(avg_regularization_length) / len(avg_regularization_length)),
		"avg_regularization_length_std": _sample_std(avg_regularization_length),
		"sparsity_reached_mean": float(sum(sparsity_reached) / len(sparsity_reached)),
		"sparsity_reached_std": _sample_std(sparsity_reached),
		"test_accuracy_mean": float(sum(test_accuracy) / len(test_accuracy)),
		"test_accuracy_std": _sample_std(test_accuracy),
	}


def main() -> None:
	args = _parse_args()
	models_root = _resolve_existing_path(args.models_root)
	output_path = _resolve_existing_path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	configure_evaluation()
	_, _, test_loader = fetch_cifar10()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	config_dirs = _discover_config_dirs(models_root)
	rows = [
		_summarize_config(config_dir.name, config_dir, device, test_loader, args.target_sparsity)
		for config_dir in config_dirs
	]

	df = pd.DataFrame(rows)
	df.to_csv(output_path, index=False)
	print(f"[SAVE] Wrote pruning summary to {output_path}")

	# Now evaluate every checkpoint from the selected runs and write per-checkpoint CSV
	all_checkpoint_rows = []
	for config_dir in config_dirs:
		config_name = config_dir.name
		run_dirs = _discover_run_dirs(config_dir)
		for run_dir in run_dirs:
			checkpoints = sorted(run_dir.glob("checkpoint_*.pth"))
			if not checkpoints:
				print(f"[WARN] No checkpoints found in {run_dir}")
				continue
			for cp in checkpoints:
				try:
					model, _ = load_model_from_checkpoint(cp, device)
					_, acc = test(model, test_loader)
					spars = calculate_real_sparsity(model)
					avg_jacobian_norm = compute_jacobian_norm(model, test_loader, device)
					condition_number_per_layer = compute_layer_condition_numbers(model)
					all_checkpoint_rows.append({"configuration": config_name, "checkpoint": str(cp), "sparsity": float(spars), "accuracy": float(acc), "avg_jacobian_norm": float(avg_jacobian_norm)})
					all_checkpoint_rows[-1].update(condition_number_per_layer)
				except Exception as e:
					print(f"[ERROR] Failed to evaluate checkpoint {cp}: {e}")

	checkpoint_out = output_path.parent / "resnet20_pruning_all_checkpoints.csv"
	df2 = pd.DataFrame(all_checkpoint_rows)
	df2.to_csv(checkpoint_out, index=False)
	print(f"[SAVE] Wrote per-checkpoint results to {checkpoint_out}")


if __name__ == "__main__":
	main()
