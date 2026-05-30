from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_SPARSITY_PATTERN = re.compile(r"sparsity_(?P<sparsity>[0-9]+(?:\.[0-9]+)?)")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Visualize saved Fourier sensitivity heatmaps for one run")
	parser.add_argument("--run-dir", type=Path, default=Path("results") / "fourier_sensitivity" / "run_03", help="Directory containing saved heatmaps")
	parser.add_argument("--output-path", type=Path, default=Path("results") / "fourier_sensitivity" / "run_03" / "fourier_dff_sensitivity_summary.png", help="Path for the combined figure")
	parser.add_argument("--metadata", type=Path, default=None, help="Optional metadata CSV written by the evaluation script")
	return parser.parse_args()


def _discover_heatmaps(run_dir: Path) -> pd.DataFrame:
	metadata_rows: list[dict[str, object]] = []

	for npy_path in sorted(run_dir.rglob("*_difference_heatmap.npy")):
		match = _SPARSITY_PATTERN.search(npy_path.stem)
		sparsity = float(match.group("sparsity")) if match else np.nan
		metadata_rows.append(
			{
				"checkpoint_name": npy_path.stem.replace("_difference_heatmap", ""),
				"checkpoint_path": str(npy_path),
				"sparsity": sparsity,
			}
		)

	if not metadata_rows:
		raise FileNotFoundError(f"No heatmaps found under {run_dir}")

	frame = pd.DataFrame(metadata_rows)
	frame = frame.sort_values(["sparsity", "checkpoint_name"], na_position="last").reset_index(drop=True)
	return frame


def _load_metadata(metadata_path: Path) -> pd.DataFrame:
	frame = pd.read_csv(metadata_path)
	if "heatmap_npy" not in frame.columns:
		raise ValueError(f"Metadata file {metadata_path} does not contain a heatmap_npy column")
	if "sparsity" not in frame.columns:
		raise ValueError(f"Metadata file {metadata_path} does not contain a sparsity column")
	frame = frame.copy()
	frame["checkpoint_name"] = frame["checkpoint_path"].map(lambda value: Path(str(value)).stem.replace("_difference_heatmap", "")) if "checkpoint_path" in frame.columns else frame["heatmap_npy"].map(lambda value: Path(str(value)).stem.replace("_difference_heatmap", ""))
	frame["checkpoint_path"] = frame["heatmap_npy"]
	return frame.sort_values(["sparsity", "checkpoint_name"], na_position="last").reset_index(drop=True)


def _select_representative_rows(frame: pd.DataFrame, target_count: int = 5) -> pd.DataFrame:
	if target_count < 1:
		raise ValueError("target_count must be at least 1")

	ordered = frame.sort_values(["sparsity", "checkpoint_name"], na_position="last").reset_index(drop=True)
	if ordered.empty:
		raise ValueError("No heatmaps available for plotting")

	dense_candidates = ordered[np.isclose(ordered["sparsity"].to_numpy(dtype=float), 0.0, atol=1e-6)]
	if dense_candidates.empty:
		raise ValueError("Could not find a 0% sparsity checkpoint to use as the dense baseline")

	dense_row = dense_candidates.iloc[0]
	remaining = ordered.drop(index=dense_row.name).reset_index(drop=True)

	if target_count == 1 or remaining.empty:
		selected = pd.DataFrame([dense_row])
		return selected.reset_index(drop=True)

	remaining_count = target_count - 1
	indices = np.linspace(0, len(remaining) - 1, remaining_count)
	indices = np.round(indices).astype(int)
	indices = np.unique(indices)
	while len(indices) < remaining_count:
		for candidate in range(len(remaining)):
			if candidate not in indices:
				indices = np.append(indices, candidate)
				if len(indices) == remaining_count:
					break

	selected_remaining = remaining.iloc[np.sort(indices[:remaining_count])]
	selected = pd.concat([pd.DataFrame([dense_row]), selected_remaining], ignore_index=True)
	return selected.reset_index(drop=True)


def _load_heatmap(path: str | Path) -> np.ndarray:
	array = np.load(path)
	if array.ndim != 2:
		raise ValueError(f"Expected a 2D heatmap in {path}, got shape {array.shape}")
	return array


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
	minimum = float(np.min(heatmap))
	maximum = float(np.max(heatmap))
	span = maximum - minimum
	if span <= 0:
		return np.zeros_like(heatmap, dtype=float)
	return (heatmap - minimum) / span


def _plot_summary(selected: pd.DataFrame, output_path: Path) -> None:
	heatmaps = [_normalize_heatmap(_load_heatmap(path)) for path in selected["checkpoint_path"]]
	dense_heatmap = heatmaps[0]
	differences = [heatmap - dense_heatmap for heatmap in heatmaps]

	first_row_min = min(float(np.min(heatmap)) for heatmap in heatmaps)
	first_row_max = max(float(np.max(heatmap)) for heatmap in heatmaps)
	delta_bound = max(abs(float(np.min(delta))) for delta in differences)
	delta_bound = max(delta_bound, max(abs(float(np.max(delta))) for delta in differences))
	if delta_bound == 0:
		delta_bound = 1.0

	fig, axes = plt.subplots(2, len(heatmaps), figsize=(4 * len(heatmaps), 8), constrained_layout=True)
	if len(heatmaps) == 1:
		axes = np.array([[axes[0]], [axes[1]]])

	for column_index, row in selected.iterrows():
		heatmap = heatmaps[column_index]
		delta = differences[column_index]
		sparsity = float(row["sparsity"])
		label = f"{sparsity:.2f}% sparsity"

		image = axes[0, column_index].imshow(
			heatmap,
			cmap="magma",
			vmin=first_row_min,
			vmax=first_row_max,
			interpolation="nearest",
		)
		axes[0, column_index].set_title(label)
		axes[0, column_index].axis("off")

		axes[1, column_index].imshow(
			delta,
			cmap="coolwarm",
			vmin=-delta_bound,
			vmax=delta_bound,
			interpolation="nearest",
		)
		axes[1, column_index].set_title(f"{label} - dense")
		axes[1, column_index].axis("off")

	fig.colorbar(image, ax=axes[0, :].tolist(), shrink=0.82, label="Normalized sensitivity (per-heatmap min-max)")
	fig.colorbar(axes[1, 0].images[0], ax=axes[1, :].tolist(), shrink=0.82, label="Difference from dense baseline")
	#fig.suptitle("Normalized Fourier sensitivity heatmaps and dense-baseline deltas", y=1.02)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=220, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	args = _parse_args()
	if args.metadata is not None:
		frame = _load_metadata(args.metadata)
	else:
		frame = _discover_heatmaps(args.run_dir)

	selected = _select_representative_rows(frame, target_count=6)
	_plot_summary(selected, args.output_path)
	print(f"[SAVE] Wrote summary figure to {args.output_path}")


if __name__ == "__main__":
	main()