# Execution plan for the adaptation of IRWL1 method for sparse architecture learning

## Phase 1: Batch Normalization

  - How do training speed and test accuracy change when applying batch normalization on every hidden layer of LeNet-5?
  - How does batch normalization impact the sparsity-accuracy curve corresponding to l1/irwl1 regularization?
  - How can the scale parameter of batchnorm be used for structured pruning (i.e. network slimming)?

## Phase 2: Gradual Pruning
  - Refine pruning method by iteratively prune only 20-10% of weights
  - Introduce warm-up at the beginnig, recovery phase after pruning step
  - Does gradually decreasing the pruning rate help?
  - Tuning the length of warm-up phase

## Phase 3: Update Interval tuning
  - Define 1D metric out of sparsity-accuracy curve to tune frequency of penalty updates using Optuna

## Phase 4: Weight re-initialization
  - After the pruning phase, re-initialize the remaining weights
  - Measure its impact on the sparsity-accuracy curve
