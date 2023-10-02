import os
from os.path import join, dirname

import torch


def save_checkpoint(state, save_dir, filename="checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    ckp_path = join(save_dir, filename)
    os.makedirs(dirname(ckp_path), exist_ok=True)
    torch.save(state, ckp_path)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer,
