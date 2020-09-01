import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD

from models.CNet import CNet 
# from losses import Loss_SceneFlow_SelfSup, Loss_SceneFlow_SemiSup
from loss_consistency import Loss_SceneFlow_SelfSup_Consistency
from augmentations import Augmentation_SceneFlow, Augmentation_SceneFlow_Carla
from datasets.kitti_raw_monosf import CarlaDataset, KITTI_Raw_KittiSplit_Train, KITTI_Raw_KittiSplit_Valid

from torchsummary import summary
from time import time

parser = argparse.ArgumentParser(description="Self Supervised Joint Learning of Scene Flow and Motion Segmentation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset_name', default='KITTI', help='KITTI or Carla')
parser.add_argument('--data_root', help='path to dataset', required=True)
# parser.add_argument('--kitti_2015_root', help='path to dataset', required=True)
parser.add_argument('--finetuning', type=bool, default=False, help='finetuning on supervised data')
parser.add_argument('--evaluation', type=bool, default=False, help='evaluating on data')
parser.add_argument('--num-views', type=int, default=2, help="number of views present in training data")
parser.add_argument('--exp_name', type=str, default='test', help='name of experiment, chkpts stored in checkpoints/experiment')
parser.add_argument('--resume_from_epoch', default=0, help='resume from checkpoint (using experiment name)')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to run')
parser.add_argument('--cuda', type=bool, default=True, help='use gpu')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd or alpha param for adam')
parser.add_argument('--beta', type=float, default=0.999, help='beta param for adam')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--dropout', type=bool, default=False, help='dropout for regularization', choices=[True, False])
parser.add_argument('--num_pyramid_levels', type=int, default=6, help='number of pyramid feature levels')
parser.add_argument('--seed', default=123, help='random seed for reproducibility')

args  = parser.parse_args()

def main():
  DATASET_NAME = args.dataset_name
  DATA_ROOT = args.data_root

  # load the dataset/dataloader
  if DATASET_NAME == 'KITTI':
    train_dataset = KITTI_Raw_KittiSplit_Train(args, DATA_ROOT)
    val_dataset = KITTI_Raw_KittiSplit_Valid(args, DATA_ROOT)
    augmentations = Augmentation_SceneFlow(args)
    loss = Loss_SceneFlow_SelfSup_Consistency(args)
  elif DATASET_NAME == 'CARLA':
    train_dataset = CarlaDataset(args, DATA_ROOT)
    val_dataset = None
    augmentations = Augmentation_SceneFlow_Carla(args)
    loss = Loss_SceneFlow_SemiSup(args)
  else:
    raise Exception('Dataset name is not valid')

  train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
  val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=8, pin_memory=True) if val_dataset else None

  lr_scheduler = None

  # load the model
  model = CNet(args)
  if args.cuda:
    model = model.cuda()

  # summary(model, None)

  params = model.parameters()
  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

  optimizer = Adam(model.parameters(), lr=args.lr, betas=[args.momentum, args.beta], weight_decay=args.weight_decay)

  for i, data in enumerate(train_dataloader):
    loss_dict, _ = step(args, data, model, loss, augmentations, optimizer)
    break

  # run training loop
  # for epoch in range(args.resume_from_epoch, args.total_epochs + 1):
    # print(f"Training epoch: {epoch + 1}")
    # loss_dict = train_one_epoch(model, train_dataloader, optimizer, augmentations)

    # do ya logging bbg0rl

    # if lr_scheduler is not None:
    #   lr_scheduler.step(epoch)

  return

def step(args, data_dict, model, loss, augmentations, optimizer):
  start = time()
  # Get input and target tensor keys
  input_keys = list(filter(lambda x: "input" in x, data_dict.keys()))
  target_keys = list(filter(lambda x: "target" in x, data_dict.keys()))
  tensor_keys = input_keys + target_keys

  # Possibly transfer to Cuda
  if args.cuda:
    for k, v in data_dict.items():
      if k in tensor_keys:
        data_dict[k] = v.cuda(non_blocking=True)

  if augmentations is not None:
    with torch.no_grad():
      data_dict = augmentations(data_dict)
  
  for k, t in data_dict.items():
    if k in input_keys:
      data_dict[k] = t.requires_grad_(True)
    if k in target_keys:
      data_dict[k] = t.requires_grad_(False)

  # optimizer.zero_grad()
  output_dict = model(data_dict)
  loss_dict = loss(output_dict, data_dict)

  print(loss_dict['cons_dict']['ego'].data)
  print(loss_dict['cons_dict']['cm'].data)

  # assert (not torch.isnan(training_loss.item())), "training_loss is NaN"

  return loss_dict, output_dict

def train_one_epoch(model, loss, dataloader, optimizer, augmentations):

  model.train()

  for i, data in enumerate(dataloader):
    step_loss = step(model, data, augmentations)

    # caclulate gradients and then do Adam step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    continue

  return

if __name__ == '__main__':
  main()