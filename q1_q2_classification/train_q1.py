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
    args = ARGS(
        epochs=5,            # 5 epochs as targeted
        inp_size=64,         # Q1: 64x64 resolution
        use_cuda=True,       # use GPU if available
        val_every=70,        # validate every 70 steps (about one "epoch" on trainval split)
        lr=1e-2,             # Adam default, good for small nets
        batch_size=128,      # 128 is good compromise for speed/memory
        step_size=2,         # decay learning rate every 2 epochs
        gamma=0.5            # decay LR by half
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
