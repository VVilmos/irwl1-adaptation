from pathlib import Path
import argparse
import random

import numpy as np
import torch

import irwl1.config as config
from irwl1.data import fetch_cifar10
from irwl1.model import ResNet20
from irwl1.regularization import L1_penalty_init
from irwl1.utils import calculate_real_sparsity, global_pruning, init_mask, train


SPARSITY_STEP = 5.0
MAX_SPARSITY = 95.0


def _configure_training() -> None:
	config.WEIGHT_PRUNING_THRESHOLD = 1e-5
	config.EPSILON = 1e-6
	config.UPDATE_PER_EPOCH = 1
	config.NUM_REG_EPOCHS = 50
	config.NUM_PRETRAIN_EPOCHS = 10
	config.NUM_RECOVERY_EPOCHS = 10
	config.MODE = "weight-wise"
	config.REG_TYPE = "WL1"
	config.MODEL = "ResNet20"
	config.WANDB_MODE = "offline"
	config.FAB_STEPS = 5
	config.WEIGHT_DECAY = 0.0


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train one ResNet20 pruning run")
	parser.add_argument("--seed", type=int, required=True, help="Random seed for initialization and data order")
	parser.add_argument("--run-id", type=int, required=True, help="Unique run id for output paths and logging")
	return parser.parse_args()


def _set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)


def _build_model(device: torch.device) -> torch.nn.Module:
	model = ResNet20().to(device)
	L1_penalty_init(model)
	init_mask(model)
	return model


def _save_checkpoint(output_dir: Path, checkpoint_index: int, sparsity: float, model: torch.nn.Module, phase: str) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_path = output_dir / f"checkpoint_{checkpoint_index:02d}_sparsity_{sparsity:.2f}.pth"
	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"phase": phase,
			"checkpoint_index": checkpoint_index,
			"sparsity": sparsity,
		},
		checkpoint_path,
	)
	return checkpoint_path


def _train_single_run(run_id: int, train_loader, val_loader, device: torch.device) -> None:
	run_dir = Path("models") / f"run_{run_id:02d}"
	run_dir.mkdir(parents=True, exist_ok=True)
	print(f"[RUN {run_id}] Starting in {run_dir}")

	model = _build_model(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

	# WARMUP
	config.EPOCHS = config.NUM_PRETRAIN_EPOCHS
	model, wandb_run = train(
		model,
		train_loader,
		val_loader,
		optimizer=optimizer,
		apply_reg=False,
		is_new_run=True,
		run_name=f"iterL1_run_{run_id:02d}",
		weight_decay=False,
	)

	sparsity = calculate_real_sparsity(model)
	_save_checkpoint(run_dir, 0, sparsity, model, "warmup")
	previous_saved_sparsity = sparsity
	checkpoint_index = 1

	while sparsity <= MAX_SPARSITY:

		print(f"[RUN {run_id}] Regularization")
		config.EPOCHS = config.NUM_REG_EPOCHS
		model, wandb_run = train(
			model,
			train_loader,
			val_loader,
			optimizer=optimizer,
			apply_reg=True,
			is_new_run=False,
			run_name=f"iterL1_run_{run_id:02d}",
			run=wandb_run,
			weight_decay=False,
		)




		print(f"[RUN {run_id}] Pruning")
		global_pruning(model, masking=True)



		print(f"[RUN {run_id}] Recovery")
		config.EPOCHS = config.NUM_RECOVERY_EPOCHS
		optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
		model, wandb_run = train(
			model,
			train_loader,
			val_loader,
			optimizer=optimizer,
			apply_reg=False,
			is_new_run=False,
			run_name=f"iterL1_run_{run_id:02d}",
			run=wandb_run,
			weight_decay=False,
		)

		sparsity = calculate_real_sparsity(model)
		if sparsity - previous_saved_sparsity >= SPARSITY_STEP or sparsity >= MAX_SPARSITY:
			_save_checkpoint(run_dir, checkpoint_index, sparsity, model, "prune_recover")
			previous_saved_sparsity = sparsity
			checkpoint_index += 1

		if sparsity >= MAX_SPARSITY:
			break

	if wandb_run is not None:
		try:
			import wandb

			wandb.finish()
		except Exception:
			pass


def main() -> None:
	args = _parse_args()
	_set_seed(args.seed)

	print("[SETUP] Loading data and model")
	train_loader, val_loader, _ = fetch_cifar10()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	_configure_training()
	Path("models").mkdir(parents=True, exist_ok=True)
	Path("results").mkdir(parents=True, exist_ok=True)

	_train_single_run(args.run_id, train_loader, val_loader, device)


if __name__ == "__main__":
	main()
