import os
import time

import torch

from irwl1.regularization import L1_penalty_init
from irwl1.data import fetch_cifar10, fetch_cifar10_test_mini
import irwl1.config as config
from irwl1.model import ResNet20
from irwl1.utils import train, init_mask, global_pruning, calculate_real_sparsity, save_sparacc_curve, test, save_cifar10c_row
from irwl1.robust import deepfool_norm, evaluate_cifar10c_per_corruption, pgd_norm, fab_norm
import pandas
from pathlib import Path


def main() -> None:
	print("[SETUP] Loading data and model")
	train_loader, val_loader, test_loader = fetch_cifar10()
	mini_test_loader = fetch_cifar10_test_mini()
	# we evaluate all corruptions under data/CIFAR-10-C using the per-corruption evaluator
	cifar10c_root = "data/CIFAR-10-C"

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = ResNet20().to(device)
	L1_penalty_init(model)
	init_mask(model)

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

	print("[WARMUP] Training without regularization")
	config.EPOCHS = config.NUM_PRETRAIN_EPOCHS
	optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
	model, wandb_run = train(
		model,
		train_loader,
		val_loader,
		optimizer=optimizer,
		apply_reg=False,
		is_new_run=True,
		run_name="iterL1",
		weight_decay=False,
	)

	spar = 0
	_, acc = test(model, test_loader)

	#corr_acc = evaluate_cifar10c_per_corruption(model, data_root=cifar10c_root, severity=3, batch_size=config.BATCH_SIZE, device=device)
	fab_acc_val = fab_norm(model, mini_test_loader)
	

	os.makedirs("models", exist_ok=True)
	spar_cp, acc_cp, pgd_norm_cp, fab_acc_cp, corr_acc_cp = [], [], [], [], []
	spar_cp.append(spar)
	acc_cp.append(acc)
	# corr_acc is a dict mapping corruption -> accuracy; save per-corruption results to a dedicated CSV
	#per_corr_csv = "results/resnet20cifar10_corruptions.csv"
	#save_cifar10c_row(spar, acc, corr_acc, path=per_corr_csv)
	# do NOT compute mean of corruption accuracies; keep placeholder for legacy curve saving
	#corr_acc_cp.append(None)
	fab_acc_cp.append(fab_acc_val)

	checkpoint_idx = 0
	while spar <= 95:
		print("[STEP] Regularization")
		config.EPOCHS = config.NUM_REG_EPOCHS
		model, wandb_run = train(
			model,
			train_loader,
			val_loader,
			optimizer=optimizer,
			apply_reg=True,
			is_new_run=False,
			run_name="iterL1",
			run=wandb_run,
			weight_decay=False,
		)

		print("[STEP] Pruning")
		global_pruning(model, masking=True)

		print("[STEP] Recovery")
		config.EPOCHS = config.NUM_RECOVERY_EPOCHS
		for param_group in optimizer.param_groups:
			param_group["lr"] = 1 * config.LEARNING_RATE
		optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
		model, wandb_run = train(
			model,
			train_loader,
			val_loader,
			optimizer=optimizer,
			apply_reg=False,
			is_new_run=False,
			run_name="iterL1",
			run=wandb_run,
			weight_decay=False,
		)

		print("[STEP] Evaluation")
		spar = calculate_real_sparsity(model)
		if (spar - spar_cp[-1] if len(spar_cp) > 0 else spar) >= 5:
			spar_cp.append(spar)
			fab_acc_val = fab_norm(model, mini_test_loader)
			#corr_acc = evaluate_cifar10c_per_corruption(model, data_root=cifar10c_root, severity=3, batch_size=config.BATCH_SIZE, device=device)
			_, test_acc = test(model, test_loader)
			acc_cp.append(test_acc)
			#save_cifar10c_row(spar, test_acc, corr_acc, path="results/resnet20cifar10_corruptions.csv")
			#corr_acc_cp.append(None)
			fab_acc_cp.append(fab_acc_val)

			checkpoint_idx += 1
			checkpoint_path = f"models/resnet20_checkpoint_{checkpoint_idx:02d}_sparsity_{spar:.2f}.pth"
			#torch.save(model.state_dict(), checkpoint_path)
			print(f"checkpoint saved: {checkpoint_path}")

	print("[SAVE] Writing curve csv")
	save_sparacc_curve(spar_cp, acc_cp, fab_norm_cp=fab_acc_cp, path="results/resnet20cifar10.csv")

	if wandb_run is not None:
		try:
			import wandb
			wandb.finish()
		except Exception:
			pass


if __name__ == "__main__":
	main()
