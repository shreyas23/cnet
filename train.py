import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description="Self Supervised Joint Learning of Scene Flow and Motion Segmentation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_root', help='path to dataset', required=True)
parser.add_argument('--kitti_2015_root', help='path to dataset', required=True)
parser.add_argument('--exp_name', type=str, default='test', help='name of experiment, chkpts stored in checkpoints/experiment')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to run')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd or alpha param for adam')
parser.add_argument('--beta', type=float, default=0.999, help='beta param for adam')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--dropout', type=bool, default=False, help='dropout for regularization', choices=[True, False])
parser.add_argument('--num_pyramid_levels', type=int, default=6, help='number of pyramid feature levels')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint (using experiment name)')

def main():
    args  = parser.parse_args()

    # load the dataset/dataloader
    dataset = None
    dataloader = None

    # load the model
    model = None

    # load the augmentations
    augmentations = None

    # load the optimizer
    optimizer = None
    
    # run training loop
    for epoch in range(args.epochs):
      print(f"Training epoch: {epoch + 1}")
      train_one_epoch(model, dataloader, optimizer, augmentations)
      continue

    return

def train_one_epoch(model, dataloader, optimizer, augmentations):

  model.train()

  for i, data in enumerate(dataloader):
    loss = torch.scalar_tensor(0) 

    # caclulate gradients and then do Adam step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    continue

  return

if __name__ == '__main__':
  main()