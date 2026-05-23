from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

import irwl1.config as config


_CIFAR10_MEAN = (0.5, 0.5, 0.5)
_CIFAR10_STD = (0.5, 0.5, 0.5)


class CIFAR10CDataset(Dataset):
	"""Dataset for one CIFAR-10-C corruption at one severity level.

	Each CIFAR-10-C corruption file contains 50,000 images grouped into five
	consecutive severity blocks of 10,000 samples. This dataset exposes only one
	block so it can be used like a standard 10K evaluation set.
	"""

	def __init__(
		self,
		root: str | Path,
		corruption: str,
		severity: int,
		normalize: bool = True,
	) -> None:
		if severity not in {1, 2, 3, 4, 5}:
			raise ValueError("severity must be one of {1, 2, 3, 4, 5}")

		self.root = Path(root)
		self.corruption = corruption.replace(".npy", "")
		self.severity = severity
		self.normalize = normalize

		corruption_file = self.root / f"{self.corruption}.npy"
		if not corruption_file.exists():
			matches = list(self.root.rglob(f"{self.corruption}.npy"))
			if not matches:
				raise FileNotFoundError(f"Could not find {self.corruption}.npy under {self.root}")
			corruption_file = matches[0]

		label_file = _find_labels_file(self.root)
		self._images = np.load(corruption_file, mmap_mode="r")
		self._labels = np.load(label_file)

		if self._images.ndim != 4:
			raise ValueError(f"Expected a 4D CIFAR-10-C array, got shape {tuple(self._images.shape)}")

		severity_count = 5
		if self._images.shape[0] % severity_count != 0:
			raise ValueError(
				f"CIFAR-10-C file {corruption_file} does not split evenly into {severity_count} severities"
			)

		chunk_size = self._images.shape[0] // severity_count
		start = (severity - 1) * chunk_size
		stop = severity * chunk_size
		self._images = self._images[start:stop]
		self._labels = self._labels[start:stop]

		if len(self._labels) != len(self._images):
			raise ValueError(
				f"Labels in {label_file} do not match {corruption_file}: {len(self._labels)} labels for {len(self._images)} images"
			)

	def __len__(self) -> int:
		return len(self._images)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
		image = torch.from_numpy(np.array(self._images[index], copy=True))
		if image.ndim != 3:
			raise ValueError(f"Expected a 3D CIFAR-10-C image, got shape {tuple(image.shape)}")
		if image.shape[-1] in {1, 3}:
			image = image.permute(2, 0, 1)
		elif image.shape[0] not in {1, 3}:
			raise ValueError(f"Could not infer channel dimension for CIFAR-10-C image with shape {tuple(image.shape)}")
		image = image.float()
		label = torch.tensor(int(self._labels[index]), dtype=torch.long)

		if self.normalize:
			mean = torch.tensor(_CIFAR10_MEAN, device=image.device).view(-1, 1, 1)
			std = torch.tensor(_CIFAR10_STD, device=image.device).view(-1, 1, 1)
			if image.max().item() > 1.5:
				image = image / 255.0
			image = (image - mean) / std

		return image, label


def _resolve_device(device: torch.device | str | None = None) -> torch.device:
	if device is None:
		return config.DEVICE
	if isinstance(device, torch.device):
		return device
	return torch.device(device)


def _normalise_cifar10_batch(inputs: torch.Tensor) -> torch.Tensor:
	inputs = inputs.float()
	if inputs.max().item() > 1.5:
		inputs = inputs / 255.0

	mean = torch.tensor(_CIFAR10_MEAN, device=inputs.device).view(1, -1, 1, 1)
	std = torch.tensor(_CIFAR10_STD, device=inputs.device).view(1, -1, 1, 1)
	return (inputs - mean) / std


def _as_nchw_tensor(array: np.ndarray) -> torch.Tensor:
	tensor = torch.from_numpy(np.asarray(array))

	if tensor.ndim != 4:
		raise ValueError(f"Expected a 4D CIFAR-10-C array, got shape {tuple(tensor.shape)}")

	if tensor.shape[-1] in {1, 3}:
		tensor = tensor.permute(0, 3, 1, 2)
	elif tensor.shape[1] not in {1, 3}:
		raise ValueError(f"Could not infer channel dimension for CIFAR-10-C array with shape {tuple(tensor.shape)}")

	return tensor


def _find_labels_file(root: Path) -> Path:
	label_candidates = sorted(
		path for path in root.rglob("*.npy") if "label" in path.stem.lower()
	)
	if not label_candidates:
		raise FileNotFoundError(
			f"Could not find a labels.npy file under {root}. Expected the standard CIFAR-10-C layout."
		)
	return label_candidates[0]


def _collect_cifar10c_files(root: Path, corruptions: Sequence[str] | None = None) -> list[Path]:
	excluded = {"label", "labels", "y_test", "test_labels"}
	files = [
		path
		for path in sorted(root.rglob("*.npy"))
		if not any(token in path.stem.lower() for token in excluded)
	]

	if corruptions is None:
		return files

	wanted = {corruption.replace(".npy", "").lower() for corruption in corruptions}
	return [path for path in files if path.stem.lower() in wanted]


def _iter_cifar10c_batches(
	root: Path,
	batch_size: int,
	corruptions: Sequence[str] | None = None,
	severity: int | None = None,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
	if severity is not None and severity not in {1, 2, 3, 4, 5}:
		raise ValueError("severity must be one of {1, 2, 3, 4, 5} or None")

	label_file = _find_labels_file(root)
	base_labels = np.load(label_file)

	for corruption_file in _collect_cifar10c_files(root, corruptions):
		images = np.load(corruption_file, mmap_mode="r")

		if severity is not None:
			severity_count = 5
			if images.shape[0] % severity_count != 0:
				raise ValueError(
					f"CIFAR-10-C file {corruption_file} does not split evenly into {severity_count} severities"
				)
			chunk_size = images.shape[0] // severity_count
			start = (severity - 1) * chunk_size
			stop = severity * chunk_size
			images = images[start:stop]

		labels = base_labels
		if len(labels) != len(images):
			if len(images) % len(labels) != 0:
				raise ValueError(
					f"Labels in {label_file} do not match {corruption_file}: {len(labels)} labels for {len(images)} images"
				)
			repeat_factor = len(images) // len(labels)
			labels = np.tile(labels, repeat_factor)

		for start_index in range(0, len(images), batch_size):
			stop_index = min(start_index + batch_size, len(images))
			batch_images = _as_nchw_tensor(images[start_index:stop_index]).float()
			batch_labels = torch.from_numpy(np.asarray(labels[start_index:stop_index])).long()
			yield batch_images, batch_labels


def _evaluate_top1_accuracy(model: torch.nn.Module, batch_iterator: Iterable[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> float:
	model.to(device)
	model_was_training = model.training
	model.eval()

	total_correct = 0
	total_examples = 0

	try:
		with torch.no_grad():
			for inputs, targets in batch_iterator:
				inputs = inputs.to(device)
				targets = targets.to(device)

				logits = model(inputs)
				predictions = logits.argmax(dim=1)

				total_correct += (predictions == targets).sum().item()
				total_examples += targets.size(0)
	finally:
		if model_was_training:
			model.train()

	if total_examples == 0:
		return 0.0

	return 100.0 * total_correct / total_examples


def create_cifar10c_dataloader(
	data_root: str | Path,
	corruption: str,
	severity: int,
	batch_size: int | None = None,
	normalize: bool = True,
	shuffle: bool = False,
	num_workers: int = 0,
) -> torch.utils.data.DataLoader:
	"""Create a DataLoader for a single CIFAR-10-C corruption at one severity.

	This should be called once during initialization and the returned DataLoader
	reused across multiple evaluations for efficiency.
	"""
	from torch.utils.data import DataLoader

	dataset = CIFAR10CDataset(
		root=data_root,
		corruption=corruption,
		severity=severity,
		normalize=normalize,
	)

	effective_batch_size = batch_size or config.BATCH_SIZE
	return DataLoader(
		dataset,
		batch_size=effective_batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=True,
	)


def evaluate_cifar10c_top1_accuracy(
	model: torch.nn.Module,
	data_root: str | Path | torch.utils.data.DataLoader = "data/CIFAR-10-C",
	batch_size: int | None = None,
	corruptions: Sequence[str] | None = None,
	severity: int | None = None,
	device: torch.device | str | None = None,
) -> float:
	"""Return the top-1 accuracy of a model on CIFAR-10-C.

	Accepts either a path (for single-use evaluation) or a pre-created DataLoader
	(recommended for repeated evaluations during training).

	Example for repeated evaluations:
	    loader = create_cifar10c_dataloader('data/CIFAR-10-C', 'fog', severity=1)
	    for epoch in range(num_epochs):
	        # train...
	        accuracy = evaluate_cifar10c_top1_accuracy(model, loader)
	"""

	resolved_device = _resolve_device(device)

	if isinstance(data_root, torch.utils.data.DataLoader):
		return _evaluate_top1_accuracy(model, data_root, resolved_device)

	root = Path(data_root)
	effective_batch_size = batch_size or config.BATCH_SIZE

	def batch_iterator() -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
		for inputs, targets in _iter_cifar10c_batches(
			root=root,
			batch_size=effective_batch_size,
			corruptions=corruptions,
			severity=severity,
		):
			yield _normalise_cifar10_batch(inputs), targets

	return _evaluate_top1_accuracy(model, batch_iterator(), resolved_device)


def evaluate_cifar10c_per_corruption(
	model: torch.nn.Module,
	data_root: str | Path = "data/CIFAR-10-C",
	severity: int = 1,
	batch_size: int | None = None,
	device: torch.device | str | None = None,
	corruptions: Sequence[str] | None = None,
	num_workers: int = 0,
) -> dict:
	"""Evaluate model on each corruption file under `data_root`.

	Returns a dict mapping corruption filename (without .npy) to top-1 accuracy (percentage).
	"""
	resolved_device = _resolve_device(device)

	root = Path(data_root)
	effective_batch_size = batch_size or config.BATCH_SIZE

	results: dict = {}

	# collect files filtered by optional `corruptions`
	files = _collect_cifar10c_files(root, corruptions)

	model_was_training = model.training
	try:
		for corruption_file in files:
			corruption = corruption_file.stem
			loader = create_cifar10c_dataloader(
				data_root=root,
				corruption=corruption,
				severity=severity,
				batch_size=effective_batch_size,
				normalize=True,
				shuffle=False,
				num_workers=num_workers,
			)

			acc = _evaluate_top1_accuracy(model, loader, resolved_device)
			results[corruption] = acc
	finally:
		if model_was_training:
			model.train()

	return results



def fab_norm(
	model: torch.nn.Module,
	data_loader: torch.utils.data.DataLoader,
	device: torch.device | str | None = None,
) -> float:
	"""Return the top-1 accuracy of the model on FAB adversarial inputs."""

	try:
		import torchattacks
	except ImportError as exc:  # pragma: no cover - dependency guard
		raise ImportError(
			"torchattacks is required for FAB evaluation. Install it with `pip install torchattacks`."
		) from exc

	resolved_device = _resolve_device(device)
	model.to(resolved_device)
	model_was_training = model.training
	model.eval()
	eps = 8 / 255
	steps = config.FAB_STEPS
	n_restarts = 1
	alpha_max = 0.1
	eta = 1.05
	beta = 0.9
	seed = 0

	attack = torchattacks.FAB(
		model,
		norm="Linf",
		eps=eps,
		steps=steps,
		n_restarts=n_restarts,
		alpha_max=alpha_max,
		eta=eta,
		beta=beta,
		verbose=False,
		seed=seed,
	)

	total_correct = 0
	total_examples = 0

	try:
		for inputs, targets in data_loader:
			inputs = inputs.to(resolved_device)
			targets = targets.to(resolved_device)

			adversarial_inputs = attack(inputs, targets)

			with torch.no_grad():
				logits = model(adversarial_inputs)
				predictions = logits.argmax(dim=1)

			total_correct += (predictions == targets).sum().item()
			total_examples += targets.size(0)
	finally:
		if model_was_training:
			model.train()

	if total_examples == 0:
		return 0.0

	return 100.0 * total_correct / total_examples


def fab_norm_in_memory(
	model: torch.nn.Module,
	images: torch.Tensor,
	labels: torch.Tensor,
	device: torch.device | str | None = None,
) -> float:
	"""Return the top-1 accuracy of the model on FAB adversarial in-memory inputs."""

	try:
		import torchattacks
	except ImportError as exc:  # pragma: no cover - dependency guard
		raise ImportError(
			"torchattacks is required for FAB evaluation. Install it with `pip install torchattacks`."
		) from exc

	resolved_device = _resolve_device(device)
	model.to(resolved_device)
	model_was_training = model.training
	model.eval()
	eps = 8 / 255
	steps = 10
	n_restarts = 1
	alpha_max = 0.1
	eta = 1.05
	beta = 0.9
	seed = 0

	attack = torchattacks.FAB(
		model,
		norm="Linf",
		eps=eps,
		steps=steps,
		n_restarts=n_restarts,
		alpha_max=alpha_max,
		eta=eta,
		beta=beta,
		verbose=False,
		seed=seed,
	)

	images = images.to(resolved_device).float()
	labels = labels.to(resolved_device).long()

	try:
		adversarial_images = attack(images, labels)
		with torch.no_grad():
			logits = model(adversarial_images)
			predictions = logits.argmax(dim=1)
		return 100.0 * (predictions == labels).float().mean().item()
	finally:
		if model_was_training:
			model.train()


def cifar10c_accuracy(
	model: torch.nn.Module,
	data_root: str | Path | torch.utils.data.DataLoader = "data/CIFAR-10-C",
	batch_size: int | None = None,
	corruptions: Sequence[str] | None = None,
	severity: int | None = None,
	device: torch.device | str | None = None,
) -> float:
	"""Shortened alias for evaluate_cifar10c_top1_accuracy."""
	return evaluate_cifar10c_top1_accuracy(
		model=model,
		data_root=data_root,
		batch_size=batch_size,
		corruptions=corruptions,
		severity=severity,
		device=device,
	)
