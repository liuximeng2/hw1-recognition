import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import random

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    ##################################################################
    # Create hyperparameter argument class with filled-in values
    # Use image size of 64x64 in Q1. Targeting mAP ~22 in 5 epochs
    ##################################################################
    # Sweep (5 epochs, 64x64): lr=1e-3/bs=64 -> 0.24; lr=1e-3/bs=32 -> 0.24;
    # lr=2e-3/bs=64 with decay at epoch 3 -> 0.22; lr=5e-4/bs=32 no decay -> 0.26.
    # Smaller batches give more Adam steps; 1e-2 diverged (~0.06); decaying
    # too early cut accuracy. step_size=5 means StepLR never fires in 5 epochs.
    args = ARGS(
        epochs=5,            # 5 epochs as targeted
        inp_size=64,         # Q1: 64x64 resolution
        use_cuda=True,       # use GPU if available (CUDA or MPS)
        val_every=80,        # ~half an epoch at batch 32 (157 steps/epoch)
        log_every=50,        # denser loss curve than the default 100
        lr=5e-4,             # more stable than 1e-3/1e-2 for this small CNN
        batch_size=32,       # more updates per epoch than 64/128
        step_size=5,         # do not decay LR during these 5 epochs
        gamma=0.5            # unused while step_size >= epochs
    )
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

    print(args)

    # initializes the model
    model = SimpleCNN(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=64, c_dim=3)
    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)
