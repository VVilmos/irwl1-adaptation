# Execution plan for the adaptation of IRWL1 method for sparse architecture learning

## Phase 1: Batch Normalization

  - How do training speed and test accuracy change when applying batch normalization on every hidden layer of LeNet-5?
     - Test Accuracy gain ~ +0.2%
  - How does batch normalization impact the sparsity-accuracy curve corresponding to l1/irwl1 regularization? (after Phase 2.1)
  - How can the scale parameter of batchnorm be used for structured pruning (i.e. network slimming)? (after Phase 2.1)

## Phase 2: Gradual Pruning
  1. Refine pruning method by iteratively prune only 20-10% of weights
  2. Introduce warm-up at the beginnig, recovery phase after pruning step (tuning their length)
  3. Does gradually decreasing the pruning rate help? (HOW??) 

## Phase 3: Update Interval tuning
  1.  Define 1D metric out of sparsity-accuracy curve to tune frequency of penalty updates using Optuna
  - - Problem: to reach 95%, takes about **20 minutes**

## Phase 4: Weight re-initialization
  1. After the pruning phase, re-initialize the remaining weights
