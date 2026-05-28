from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from irwl1.data import fetch_cifar10, fetch_cifar10_test_mini
from irwl1.eval_utils import configure_evaluation, load_model_from_checkpoint
from irwl1.robust import evaluate_cifar10c_per_corruption, fab_norm
from irwl1.utils import test


def _evaluate_mean_corrupted_error(model, cifar10c_root: str, device: torch.device) -> dict[str, float]:
	corruption_accuracies = evaluate_cifar10c_per_corruption(model, cifar10c_root, device=device)
	mean_accuracy = sum(corruption_accuracies.values()) / len(corruption_accuracies)
	return {
		"cifar10c_mean_accuracy": mean_accuracy,
		"cifar10c_mean_error": 100.0 - mean_accuracy,
	}


def _discover_checkpoints() -> list[Path]:
	root = Path("models")
	checkpoints = sorted(root.glob("weightdecay_run_*/checkpoint_*.pth"))
	if not checkpoints:
		checkpoints = sorted(root.rglob("checkpoint_*.pth"))
	return checkpoints



def main() -> None:
	print("[SETUP] Loading evaluation data")
	_, _, test_loader = fetch_cifar10()
	mini_test_loader = fetch_cifar10_test_mini()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	cifar10c_root = "data/CIFAR-10-C"
	output_path = Path("results") / "resnet20_weightdecay_checkpoint_evaluations.csv"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	configure_evaluation()

	rows: list[dict[str, object]] = []
	checkpoints = _discover_checkpoints()
	if not checkpoints:
		raise FileNotFoundError("No checkpoints found under models/. Run scripts/resnet_train.py first.")

	for checkpoint_path in checkpoints:
		print(f"[EVAL] {checkpoint_path}")
		model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
		_, test_accuracy = test(model, test_loader)
		fab_accuracy, fab_norm_value = fab_norm(model, mini_test_loader, device=device)
		cifar10c_results = _evaluate_mean_corrupted_error(model, cifar10c_root, device)
		row = {
			"checkpoint_path": str(checkpoint_path),
			"run_dir": checkpoint_path.parent.name,
			"checkpoint_index": checkpoint.get("checkpoint_index") if isinstance(checkpoint, dict) else None,
			"phase": checkpoint.get("phase") if isinstance(checkpoint, dict) else None,
			"sparsity": checkpoint.get("sparsity") if isinstance(checkpoint, dict) else None,
			"test_accuracy": test_accuracy,
			"fab_accuracy": fab_accuracy,
			"fab_norm": fab_norm_value,
		}
		row.update(cifar10c_results)
		rows.append(row)

	df = pd.DataFrame(rows)
	df.to_csv(output_path, index=False)
	print(f"[SAVE] Wrote evaluation results to {output_path}")


if __name__ == "__main__":
	main()
