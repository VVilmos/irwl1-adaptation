from pathlib import Path

import torch

import irwl1.config as config
from irwl1.data import fetch_cifar10
from irwl1.model import ResNet20
from irwl1.regularization import L1_penalty_init
from irwl1.utils import calculate_real_sparsity, global_pruning, init_mask, train


NUM_RUNS = 1
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


def _train_single_run(run_index: int, train_loader, val_loader, device: torch.device) -> None:
	run_dir = Path("models") / f"run_{run_index:02d}"
	run_dir.mkdir(parents=True, exist_ok=True)
	print(f"[RUN {run_index + 1}/{NUM_RUNS}] Starting in {run_dir}")

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
		run_name=f"iterL1_run_{run_index:02d}",
		weight_decay=False,
	)

	sparsity = calculate_real_sparsity(model)
	_save_checkpoint(run_dir, 0, sparsity, model, "warmup")
	previous_saved_sparsity = sparsity
	checkpoint_index = 1

	while sparsity <= MAX_SPARSITY:

		print(f"[RUN {run_index + 1}/{NUM_RUNS}] Regularization")
		config.EPOCHS = config.NUM_REG_EPOCHS
		model, wandb_run = train(
			model,
			train_loader,
			val_loader,
			optimizer=optimizer,
			apply_reg=True,
			is_new_run=False,
			run_name=f"iterL1_run_{run_index:02d}",
			run=wandb_run,
			weight_decay=False,
		)




		print(f"[RUN {run_index + 1}/{NUM_RUNS}] Pruning")
		global_pruning(model, masking=True)



		print(f"[RUN {run_index + 1}/{NUM_RUNS}] Recovery")
		config.EPOCHS = config.NUM_RECOVERY_EPOCHS
		optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
		model, wandb_run = train(
			model,
			train_loader,
			val_loader,
			optimizer=optimizer,
			apply_reg=False,
			is_new_run=False,
			run_name=f"iterL1_run_{run_index:02d}",
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
	print("[SETUP] Loading data and model")
	train_loader, val_loader, _ = fetch_cifar10()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	_configure_training()
	Path("models").mkdir(parents=True, exist_ok=True)
	Path("results").mkdir(parents=True, exist_ok=True)

	for run_index in range(NUM_RUNS):
		_train_single_run(run_index, train_loader, val_loader, device)


if __name__ == "__main__":
	main()
