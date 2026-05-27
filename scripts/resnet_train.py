from pathlib import Path
import pandas as pd
import argparse
import random

import numpy as np
import torch

import irwl1.config as config
from irwl1.data import fetch_cifar10
from irwl1.model import ResNet20
from irwl1.regularization import L1_penalty_init
from irwl1.utils import calculate_real_sparsity, train_regularized, global_pruning, init_mask, train, rewind_model_to_checkpoint, zero_pruned_optimizer_state


MAX_SPARSITY = 90.0


def _configure_training() -> None:
	config.WEIGHT_PRUNING_THRESHOLD = 1e-5
	config.EPSILON = config.EPSILON_START
	config.UPDATE_PER_EPOCH = 1
	config.MODE = "weight-wise"
	config.REG_TYPE = "WL1"
	config.MODEL = "ResNet20"
	config.WANDB_MODE = "offline"
	config.WEIGHT_DECAY = 1e-4
	config.LEARNING_RATE = 0.001
	config.REWIND_EPOCH = 4


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train one ResNet20 pruning run")
	parser.add_argument("--seed", type=int, required=True, help="Random seed for initialization and data order")
	parser.add_argument("--run-id", type=int, required=True, help="Unique run id for output paths and logging")
	parser.add_argument("--rewind", dest="rewind", action="store_true", help="Enable weight rewinding")
	parser.add_argument("--no-rewind", dest="rewind", action="store_false", help="Disable weight rewinding")
	parser.add_argument("--smooth", dest="smooth", action="store_true", help="Enable smooth schedules for epsilon and lambda")
	parser.add_argument("--static", dest="smooth", action="store_false", help="Disable smooth schedules (static epsilon/lambda)")
	parser.set_defaults(rewind=config.REWIND, smooth=False)
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


def _checkpoint_folder(run_id: int, rewind: bool, smooth: bool) -> Path:
	mode_folder = f"{'smooth' if smooth else 'static'}_{'rewind' if rewind else 'no_rewind'}"
	return Path("models") / mode_folder / str(run_id)


def _train_single_run(run_id: int, train_loader, val_loader, device: torch.device, rewind: bool, smooth: bool) -> None:
	run_dir = _checkpoint_folder(run_id, rewind, smooth)
	run_dir.mkdir(parents=True, exist_ok=True)
	print(f"[RUN {run_id}] Starting in {run_dir}")
	rewind_checkpoint_path = run_dir / "rewind_checkpoint.pth"

	model = _build_model(device)
	optimizer = _build_optimizer(model)
	current_real_sparsity = calculate_real_sparsity(model)

	wandb_run = None
	total_epochs = 0
	iteration = 0
	epoch_records = []

	while True:

		model, wandb_run, epochs = train(
			model,
			train_loader,
			val_loader,
			optimizer=optimizer,
			is_new_run=(wandb_run is None),
			run_name=f"iterL1_run_{run_id:02d}",
			run=wandb_run,
			weight_decay=True,
			rewind=(rewind and (wandb_run is None)),
			rewind_checkpoint_path=rewind_checkpoint_path,
			rewind_epoch=config.REWIND_EPOCH,
		)
		total_epochs += epochs

		train_epochs = epochs
		

		current_real_sparsity = calculate_real_sparsity(model)

		state_dict = model.state_dict()
		checkpoint_path = run_dir / f"checkpoint_spar{current_real_sparsity:.2f}.pth"
		torch.save(state_dict, checkpoint_path)
		if current_real_sparsity >= MAX_SPARSITY:
			print(f"[RUN {run_id}] Reached target sparsity: {current_real_sparsity:.2f}%")
			# record iteration info (no regularization phase ran)
			epoch_records.append({"iteration": iteration, "train_epochs": train_epochs, "reg_epochs": 0, "sparsity": current_real_sparsity})
			break

		model, wandb_run, epochs = train_regularized(
			model,
			train_loader,
			val_loader,
			optimizer=optimizer,
			is_new_run=False,
			run_name=f"iterL1_run_{run_id:02d}",
			run=wandb_run,
			weight_decay=False,
		)
		reg_epochs = epochs
		total_epochs += reg_epochs

		# record iteration's epoch counts
		epoch_records.append({"iteration": iteration, "train_epochs": train_epochs, "reg_epochs": reg_epochs, "sparsity": current_real_sparsity})
		iteration += 1

		global_pruning(model, masking=True)
		if rewind:
			if rewind_checkpoint_path.exists():
				model = rewind_model_to_checkpoint(model, rewind_checkpoint_path)
				optimizer = _build_optimizer(model)
			else:
				print(f"[RUN {run_id}] Rewind checkpoint not found at {rewind_checkpoint_path}; continuing without rewinding")
				zero_pruned_optimizer_state(model, optimizer)
		else:
			zero_pruned_optimizer_state(model, optimizer)

	if wandb_run is not None:
		try:
			import wandb

			wandb.finish()
		except Exception:
			pass

	# write total epochs executed during this run to CSV in the run folder
	try:
		csv_path = run_dir / "total_epochs.csv"
		with open(csv_path, "w") as fh:
			fh.write("total_epochs\n")
			fh.write(str(total_epochs) + "\n")
	except Exception as e:
		print(f"Warning: could not write total_epochs.csv: {e}")

	# write epoch-by-iteration records
	try:
		df = pd.DataFrame(epoch_records)
		iter_csv = run_dir / "epochs_by_iteration.csv"
		df.to_csv(iter_csv, index=False)
		if wandb_run is not None:
			try:
				import wandb
				wandb_run.log({"epoch_records": wandb.Table(dataframe=df)})
			except Exception:
				pass
	except Exception as e:
		print(f"Warning: could not write epochs_by_iteration.csv: {e}")


def main() -> None:
	args = _parse_args()
	_set_seed(args.seed)

	print("[SETUP] Loading data and model")
	train_loader, val_loader, _ = fetch_cifar10()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	_configure_training()
	# Apply CLI flags to config so --smooth/--static and --rewind take effect
	config.REWIND = args.rewind
	config.IS_EPSILON_DECAY = args.smooth
	config.IS_LAMBDA_RISE = args.smooth
	Path("models").mkdir(parents=True, exist_ok=True)
	Path("results").mkdir(parents=True, exist_ok=True)

	_train_single_run(args.run_id, train_loader, val_loader, device, args.rewind, args.smooth)


if __name__ == "__main__":
	main()
