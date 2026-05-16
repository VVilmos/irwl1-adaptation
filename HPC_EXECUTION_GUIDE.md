# ResNet20 CIFAR-10 Iterative Pruning - HPC Execution Guide

Updated `scripts/resnet_train.py` is now optimized for HPC cluster execution with the following improvements:

## Key Features

✓ **Full notebook training strategy** - Matches the iterative pruning workflow from the notebook
✓ **Comprehensive logging** - File and stdout logging with timestamps
✓ **Model checkpointing** - Automatic checkpoint saving at each pruning milestone
✓ **Command-line configuration** - All parameters configurable via CLI arguments
✓ **W&B optional** - Weights & Biases logging can be disabled for HPC environments
✓ **Error handling** - Graceful handling of interrupts and errors
✓ **Device flexibility** - CPU or GPU (auto-detected by default)
✓ **Path flexibility** - All paths configurable for different cluster layouts

## Running Locally

```bash
python scripts/resnet_train.py \
    --data-root ./data \
    --output-dir ./results \
    --checkpoint-dir ./models \
    --disable-wandb
```

## Running on HPC Cluster

### Step 1: Prepare the environment

```bash
# Load modules (adjust for your cluster)
module load python/3.11
module load cuda/12.1

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare SLURM script

Edit `scripts/submit_resnet_hpc.sh`:
- Update module names for your cluster (`module load python/X.X`, `module load cuda/X.X`)
- Update virtual environment path
- Update project directory path (`cd /path/to/onlab`)
- Adjust SLURM directives:
  - `--partition`: GPU queue name for your cluster
  - `--time`: Wall-clock time limit
  - `--mem`: Memory per node
  - `--cpus-per-task`: Number of CPU threads
  - `--gres=gpu:X`: Number of GPUs

### Step 3: Submit job

```bash
sbatch scripts/submit_resnet_hpc.sh
```

Monitor job:
```bash
squeue -u $USER
tail -f logs/resnet20-<JOBID>.out
```

## Command-Line Arguments

```
--data-root PATH              Root directory for CIFAR-10 dataset (default: ./data)
--output-dir PATH             Directory for CSV results (default: ./results)
--checkpoint-dir PATH         Directory for model checkpoints (default: ./models)
--cifar10c-dir PATH           Directory for CIFAR-10-C corrupted images (default: ./data/CIFAR-10-C)
--max-sparsity FLOAT          Stop at this sparsity level (default: 95.0)
--sparsity-increment FLOAT    Min increase to save checkpoint (default: 5.0)
--deepfool-steps INT          DeepFool iteration count (default: 50, reduce for faster runs)
--num-workers INT             Data loader workers (default: 4, increase for I/O-bound clusters)
--device DEVICE               'cuda' or 'cpu' (default: auto-detect)
--disable-wandb               Disable Weights & Biases logging
```

## Example: Fast HPC run (shorter DeepFool for testing)

```bash
python scripts/resnet_train.py \
    --data-root /mnt/data/cifar10 \
    --output-dir /mnt/results \
    --checkpoint-dir /mnt/models \
    --deepfool-steps 20 \
    --num-workers 16 \
    --disable-wandb
```

## Checkpoint Management

Checkpoints are saved with descriptive names:
```
models/resnet20_checkpoint_01_sparsity_5.00.pth
models/resnet20_checkpoint_02_sparsity_10.50.pth
...
```

Load a checkpoint:
```python
model.load_state_dict(torch.load("models/resnet20_checkpoint_01_sparsity_5.00.pth"))
```

## Output Files

After training completes:
- `results/resnet20cifar10.csv` - Sparsity/accuracy/robustness curves
- `results/resnet20_YYYYMMDD_HHMMSS.log` - Detailed training log
- `models/resnet20_checkpoint_*.pth` - Model checkpoints at each milestone

## Troubleshooting

**Out of memory on GPU:**
```bash
# Reduce batch size in config or load data more efficiently
# Check config.BATCH_SIZE in irwl1/config.py
```

**DeepFool takes too long:**
```bash
# Reduce steps parameter
--deepfool-steps 20  # instead of 50
```

**CIFAR-10-C not found:**
The script will skip corrupted accuracy evaluation if dataset is missing. Download from:
https://zenodo.org/record/2535967

**W&B connection issues on HPC:**
```bash
# Use --disable-wandb flag to skip W&B
```

## Expected Runtime

Typical timeline for one full iterative pruning run (sparsity 0→95%):
- **Warm-up (10 epochs):** ~10-15 min
- **Per iteration (reg + prune + recovery):** ~20-30 min
- **Checkpoint evaluation (DeepFool + CIFAR-10-C):** ~5-10 min
- **Total (~18-20 checkpoints):** ~6-8 hours on single GPU

Use `--deepfool-steps 20` to reduce to ~3-4 hours.
