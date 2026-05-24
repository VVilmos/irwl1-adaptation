from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import irwl1.config as config
from irwl1.data import fetch_cifar10
from irwl1.model import ResNet20
from irwl1.regularization import L1_penalty_init
from irwl1.utils import init_mask, test


def _configure_evaluation() -> None:
	config.WEIGHT_PRUNING_THRESHOLD = 1e-5
	config.EPSILON = 1e-6
	config.UPDATE_PER_EPOCH = 1
	config.MODE = "weight-wise"
	config.REG_TYPE = "WL1"
	config.MODEL = "ResNet20"
	config.WANDB_MODE = "offline"
	config.FAB_STEPS = 10
	config.WEIGHT_DECAY = 0.0


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate Fourier sensitivity for all checkpoints in one run")
	parser.add_argument("--run-dir", type=Path, default=Path("models") / "run_01", help="Checkpoint directory to scan")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("results") / "fourier_sensitivity" / "run_01",
		help="Directory where heatmaps, plots, and metadata will be written",
	)
	parser.add_argument("--epsilon", type=float, default=15 / 255, help="Fourier perturbation budget")
	return parser.parse_args()


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
	model = ResNet20().to(device)
	L1_penalty_init(model)
	init_mask(model)
	checkpoint = torch.load(checkpoint_path, map_location=device)
	state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
	model.load_state_dict(state_dict)
	return model, checkpoint if isinstance(checkpoint, dict) else {}


def _discover_checkpoints(run_dir: Path) -> list[Path]:
	checkpoints = sorted(run_dir.glob("checkpoint_*.pth"))
	if checkpoints:
		return checkpoints
	return sorted(run_dir.rglob("checkpoint_*.pth"))


def generate_fourier_basis(height: int, width: int) -> torch.Tensor:
	basis_matrices = torch.zeros((height, width, height, width))

	for i in range(height):
		for j in range(width):
			freq_spectrum = torch.zeros((height, width), dtype=torch.complex64)
			freq_spectrum[i, j] = 1.0
			spatial_wave = torch.fft.ifft2(torch.fft.ifftshift(freq_spectrum)).real
			norm = torch.norm(spatial_wave, p=2)
			if norm > 0:
				spatial_wave /= norm
			basis_matrices[i, j] = spatial_wave

	return basis_matrices


def evaluate_fourier_sensitivity_ratio_optimized(
	model: torch.nn.Module,
	dataloader,
	device: torch.device,
	clean_error_rate: float,
	epsilon: float = 15 / 255,
	basis_set: torch.Tensor | None = None,
) -> np.ndarray:
	model.eval()
	model.to(device)

	height, width = 32, 32
	if basis_set is None:
		basis_set = generate_fourier_basis(height, width).to(device)
	else:
		basis_set = basis_set.to(device)

	if clean_error_rate <= 0:
		raise ValueError("clean_error_rate must be positive to compute a sensitivity ratio")

	heatmap = np.zeros((height, width), dtype=np.float32)

	print("Evaluating normalized Fourier sensitivity grid...")
	for i in tqdm(range(height), desc="Processing Frequency Rows"):
		U_row = basis_set[i]
		frequency_count = U_row.size(0)

		total_errors = torch.zeros(frequency_count, device=device)
		total_samples = 0

		for images, labels in dataloader:
			images = images.to(device)
			labels = labels.to(device)
			batch_size = images.size(0)

			rand_values = torch.randint(0, 2, (batch_size, frequency_count, 1, 1, 1), device=device)
			rand_signs = rand_values * 2.0 - 1.0
			U_expanded = U_row.unsqueeze(0).unsqueeze(2)
			perturbation = rand_signs * epsilon * U_expanded
			perturbed_images = torch.clamp(images.unsqueeze(1) + perturbation, -1.0, 1.0)
			flat_perturbed = perturbed_images.view(batch_size * frequency_count, 3, height, width)

			with torch.inference_mode():
				outputs = model(flat_perturbed)

			preds = outputs.view(batch_size, frequency_count, -1).argmax(dim=2)
			batch_errors = (preds != labels.unsqueeze(1)).sum(dim=0)
			total_errors += batch_errors
			total_samples += batch_size

		row_error_rate = (total_errors / total_samples).detach().cpu().numpy()
		heatmap[i, :] = row_error_rate / clean_error_rate

	return heatmap


def plot_fourier_heatmap(heatmap: np.ndarray, output_path: Path, title: str) -> None:
	fig, ax = plt.subplots(figsize=(6.5, 6.5))
	image = ax.imshow(heatmap, cmap="magma", interpolation="nearest")
	fig.colorbar(image, ax=ax, label="Perturbed error / clean error")
	ax.set_title(title)
	ax.set_xlabel("Horizontal Frequency (j)")
	ax.set_ylabel("Vertical Frequency (i)")
	ax.axis("off")
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	args = _parse_args()
	_configure_evaluation()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	_, _, test_loader = fetch_cifar10()
	basis_set = generate_fourier_basis(32, 32).to(device)
	checkpoints = _discover_checkpoints(args.run_dir)
	if not checkpoints:
		raise FileNotFoundError(f"No checkpoints found under {args.run_dir}")

	args.output_dir.mkdir(parents=True, exist_ok=True)
	metadata_rows: list[dict[str, object]] = []

	for checkpoint_path in checkpoints:
		print(f"[EVAL] {checkpoint_path}")
		model, checkpoint = _load_model_from_checkpoint(checkpoint_path, device)
		_, clean_accuracy = test(model, test_loader)
		clean_error_rate = 1.0 - (clean_accuracy / 100.0)
		heatmap = evaluate_fourier_sensitivity_ratio_optimized(
			model=model,
			dataloader=test_loader,
			device=device,
			clean_error_rate=clean_error_rate,
			epsilon=args.epsilon,
			basis_set=basis_set,
		)

		checkpoint_stem = checkpoint_path.stem
		checkpoint_output_dir = args.output_dir / checkpoint_stem
		checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
		npy_path = checkpoint_output_dir / f"{checkpoint_stem}_ratio_heatmap.npy"
		png_path = checkpoint_output_dir / f"{checkpoint_stem}_ratio_heatmap.png"
		np.save(npy_path, heatmap)
		plot_fourier_heatmap(
			heatmap,
			png_path,
			title=f"Fourier sensitivity ratio for {checkpoint_stem}\n(clean error = {clean_error_rate:.4f})",
		)

		metadata_rows.append(
			{
				"checkpoint_path": str(checkpoint_path),
				"checkpoint_index": checkpoint.get("checkpoint_index") if isinstance(checkpoint, dict) else None,
				"phase": checkpoint.get("phase") if isinstance(checkpoint, dict) else None,
				"sparsity": checkpoint.get("sparsity") if isinstance(checkpoint, dict) else None,
				"clean_accuracy": clean_accuracy,
				"clean_error_rate": clean_error_rate,
				"epsilon": args.epsilon,
				"heatmap_npy": str(npy_path),
				"heatmap_png": str(png_path),
			}
		)

	metadata_path = args.output_dir / "fourier_sensitivity_ratio_metadata.csv"
	pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
	print(f"[SAVE] Wrote heatmaps and metadata to {args.output_dir}")


if __name__ == "__main__":
	main()