from pathlib import Path
import argparse
import random

import numpy as np
import torch

import irwl1.config as config
from irwl1.data import fetch_cifar10
from irwl1.model import ResNet20
from irwl1.regularization import L1_penalty_init
from irwl1.utils import calculate_real_sparsity, calculate_thresholded_sparsity, global_pruning, init_mask, train


MAX_SPARSITY = 95.0
SPARSITY_STALL_PATIENCE = 9
MIN_SPARSITY_IMPROVEMENT = 0.0
MAX_NO_PROGRESS_PRUNES = 3


def _configure_training() -> None:
	config.WEIGHT_PRUNING_THRESHOLD = 1e-5
	config.EPSILON = config.EPSILON_START

	# Update penalties once per epoch and decay epsilon on the same cadence.
	config.UPDATE_PER_EPOCH = 1
	config.PATIENCE = SPARSITY_STALL_PATIENCE

	config.MODE = "weight-wise"
	config.REG_TYPE = "WL1"
	config.MODEL = "ResNet20"
	config.WANDB_MODE = "online"
	config.WEIGHT_DECAY = 1e-4
	config.LAMBDA_REG = 0.01
	config.LEARNING_RATE = 0.001


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


def _build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
	return torch.optim.Adam(
		model.parameters(),
		lr=config.LEARNING_RATE,
		weight_decay=config.WEIGHT_DECAY,
	)
def _train_single_run(run_id: int, train_loader, val_loader, device: torch.device) -> None:
	run_dir = Path("models") / f"weightdecay_run_{run_id:02d}"
	run_dir.mkdir(parents=True, exist_ok=True)
	print(f"[RUN {run_id}] Starting in {run_dir}")

	model = _build_model(device)
	optimizer = _build_optimizer(model)

	wandb_run = None

	while True:

		model, wandb_run = train(
			model,
			train_loader,
			val_loader,
			optimizer=optimizer,
			is_new_run=(wandb_run is None),
			run_name=f"iterL1_run_{run_id:02d}",
			run=wandb_run,
			weight_decay=False,
		)

		current_real_sparsity = calculate_real_sparsity(model)


		if current_real_sparsity >= MAX_SPARSITY:
			print(f"[RUN {run_id}] Reached target sparsity: {current_real_sparsity:.2f}%")
			break

		global_pruning(model, masking=True)

		# Restart cycle with weak regularization and fresh optimizer state.
		optimizer = _build_optimizer(model)

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
