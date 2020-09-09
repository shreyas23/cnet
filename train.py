import os
import argparse
import datetime
import numpy as np
from sys import exit
from time import time
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
from torchsummary import summary

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.CNet import CNet 
from utils.inverse_warp import flow_warp, pose2flow, inverse_warp
from utils.sceneflow_util import projectSceneFlow2Flow
from losses import Loss_SceneFlow_SelfSup, _generate_image_left
from models.CNet import CNet_CamMaskCombined
from models.model_monosceneflow import MonoSceneFlow
from augmentations import Augmentation_SceneFlow, Augmentation_SceneFlow_Carla
from datasets.kitti_raw_monosf import CarlaDataset, KITTI_Raw_KittiSplit_Train, KITTI_Raw_KittiSplit_Valid
from loss_consistency import Loss_SceneFlow_SelfSup_Consistency, Loss_SceneFlow_SelfSup_Consistency_Decomposed
from utils.sceneflow_util import disp2depth_kitti


parser = argparse.ArgumentParser(description="Self Supervised Joint Learning of Scene Flow and Motion Segmentation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# runtime params
parser.add_argument('--data_root', help='path to dataset', required=True)
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to run')
parser.add_argument('--start_epoch', type=int, default=0, help='resume from checkpoint (using experiment name)')
parser.add_argument('--cuda', type=bool, default=True, help='use gpu?')
parser.add_argument('--no_logging', type=bool, default=False, help="are you logging this experiment?")
parser.add_argument('--log_dir', type=str, default="/external/cnet/checkpoints", help="are you logging this experiment?")
parser.add_argument('--exp_name', type=str, default='test', help='name of experiment, chkpts stored in checkpoints/experiment')
parser.add_argument('--debugging', type=bool, default=False, help='are you debugging?')

# dataset params
parser.add_argument('--dataset_name', default='KITTI', help='KITTI or Carla')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_views', type=int, default=2, help="number of views present in training data")
parser.add_argument('--num_examples', type=int, default=-1, help="number of examples to train on per epoch")
parser.add_argument('--num_workers', type=int, default=8, help="number of workers for the dataloader")
parser.add_argument('--shuffle_dataset', type=bool, default=True, help='shuffle the dataset?')

# learning params
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd or alpha param for adam')
parser.add_argument('--beta', type=float, default=0.999, help='beta param for adam')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--dropout', type=bool, default=False, help='dropout for regularization', choices=[True, False])
parser.add_argument('--grad_clip', type=int, default=5, help='gradient clipping threshold')

# model params
parser.add_argument('--num_pyramid_levels', type=int, default=6, help='number of pyramid feature levels')
parser.add_argument('--train_consistency', type=bool, default=False, help="whether to use consistency losses in training procedure")
parser.add_argument('--mask_thresh', type=float, default=.6, help='mask threshold for moving objects (higher threshold skews towards static)')

# etc. 
parser.add_argument('--finetuning', type=bool, default=False, help='finetuning on supervised data')
parser.add_argument('--evaluation', type=bool, default=False, help='evaluating on data')
parser.add_argument('--torch_seed', default=0, help='random seed for reproducibility')
parser.add_argument('--cuda_seed', default=1, help='random seed for reproducibility')

args  = parser.parse_args()

def main():
  torch.manual_seed(args.torch_seed)
  torch.cuda.manual_seed(args.cuda_seed)

  DATASET_NAME = args.dataset_name
  DATA_ROOT = args.data_root

  # load the dataset/dataloader
  print("Loading dataset and dataloaders...")
  if DATASET_NAME == 'KITTI':
    train_dataset = KITTI_Raw_KittiSplit_Train(args, DATA_ROOT, num_examples=args.num_examples, flip_augmentations=False)
    val_num_examples = int(args.num_examples / 5) if args.num_examples > 0 else -1
    val_dataset = KITTI_Raw_KittiSplit_Valid(args, DATA_ROOT, num_examples=val_num_examples)
    val_dataset = None
    augmentations = Augmentation_SceneFlow(args, photometric=False, trans=0.00, scale=[0.999, 1.0])
    # loss = Loss_SceneFlow_SelfSup_Consistency_Decomposed(args)
    loss = Loss_SceneFlow_SelfSup(args)
  elif DATASET_NAME == 'CARLA':
    train_dataset = CarlaDataset(args, DATA_ROOT)
    val_dataset = None
    augmentations = Augmentation_SceneFlow_Carla(args)
    loss = Loss_SceneFlow_SemiSup(args)
  else:
    raise Exception('Dataset name is not valid')

  train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=args.shuffle_dataset, num_workers=args.num_workers, pin_memory=True)
  val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=8, pin_memory=True) if val_dataset else None

  # load the model
  print("Loding model and augmentations and placing on gpu...")
  # model = CNet_CamMaskCombined(args)
  model = MonoSceneFlow(args)

  if args.cuda:
    augmentations = augmentations.cuda()
    model = model.cuda()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"The model has {num_params} parameters")

  # load optimizer and lr scheduler
  optimizer = Adam(model.parameters(), lr=args.lr, betas=[args.momentum, args.beta], weight_decay=args.weight_decay)
  # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 80, 200, 300], gamma=0.1)
  lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, verbose=True, mode='min', patience=10)
  lr_scheduler=None

  # set up logging
  if not args.no_logging:
    log_dir = os.path.join(args.log_dir, args.exp_name, datetime.datetime.now().strftime("%H%M%S-%Y%m%d"))
    if not os.path.isdir(log_dir):
      os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)

  if args.start_epoch > 0:
    load_epoch = args.start_epoch - 1
    ckpt_fp = os.path.join(log_dir, f"{load_epoch}.ckpt")

    print(f"Loading model from {ckpt_fp}...")

    ckpt = torch.load(ckpt_fp)
    assert (ckpt['epoch'] == load_epoch), "epoch from state dict does not match with args"
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['opt_state_dict'])

  # run training loop
  for epoch in range(args.start_epoch, args.epochs + 1):
    print(f"Training epoch: {epoch}...")
    train_loss_avg_dict, output_dict, input_dict = train_one_epoch(args, model, loss, train_dataloader, optimizer, augmentations, lr_scheduler)
    # print(f"\t Epoch {epoch} train loss avg:")
    # pprint(train_loss_avg_dict)

    if val_dataset is not None:
      print(f"Validation epoch: {epoch}...")
      val_loss_avg = eval(args, model, loss, val_dataloader, augmentations)
      print(f"\t Epoch {epoch} val loss avg: {val_loss_avg}")

    if not args.no_logging:
      writer.add_scalar('loss/train', train_loss_avg_dict['total_loss'], epoch)
      # writer.add_scalar('loss/val', val_loss_avg, epoch)
      writer.add_scalar('loss/train/sf', train_loss_avg_dict['sf'], epoch)
      writer.add_scalar('loss/train/dp', train_loss_avg_dict['dp'], epoch)
      writer.add_scalar('loss/train/s_2', train_loss_avg_dict['s_2'], epoch)
      writer.add_scalar('loss/train/s_3', train_loss_avg_dict['s_3'], epoch)
      writer.add_scalar('loss/train/s_3s', train_loss_avg_dict['s_3s'], epoch)
      # writer.add_scalar('loss/train/sanity', train_loss_avg_dict['sanity'], epoch)

      if epoch % 20 == 0:
        visualize_output(args, input_dict, output_dict, epoch, writer)

      if args.train_consistency:
        writer.add_scalar('loss/train/cons/total', train_loss_avg_dict['cons'], epoch)
        writer.add_scalar('loss/train/cons/cm', train_loss_avg_dict['cons_dict']['cm'], epoch)
        writer.add_scalar('loss/train/cons/mask', train_loss_avg_dict['cons_dict']['mask'], epoch)
        writer.add_scalar('loss/train/cons/static', train_loss_avg_dict['cons_dict']['static'], epoch)
        writer.add_scalar('loss/train/cons/ego', train_loss_avg_dict['cons_dict']['ego'], epoch)

    assert (not torch.isnan(train_loss_avg_dict['total_loss'])), "avg training loss is nan"

    if lr_scheduler is not None:
      # lr_scheduler.step(epoch)
      lr_scheduler.step(train_loss_avg_dict['total_loss'])

    # save model
    if not args.no_logging:
      if epoch % 1000 == 0 or epoch == args.epochs:
        fp = os.path.join(log_dir, f"{epoch}.ckpt")

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'opt_state_dict': optimizer.state_dict()}, fp)

  if not args.no_logging:
    writer.flush()

  return


def step(args, data_dict, model, loss, augmentations, optimizer):
  start = time()
  # Get input and target tensor keys
  input_keys = list(filter(lambda x: "input" in x, data_dict.keys()))
  target_keys = list(filter(lambda x: "target" in x, data_dict.keys()))
  tensor_keys = input_keys + target_keys
  debug_keys = ['input_l1', 'input_l2', 'input_r1', 'input_r2']

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

  output_dict = model(data_dict)
  loss_dict = loss(output_dict, data_dict)

  training_loss = loss_dict['total_loss']
  assert (not torch.isnan(training_loss)), "training_loss is NaN"

  return loss_dict, output_dict


def train_one_epoch(args, model, loss, dataloader, optimizer, augmentations, lr_scheduler):

  keys =  ['total_loss', 'dp', 's_2', 's_3', 'sf', 's_3s']

  if args.train_consistency:
    keys.append('cons')
    cons_keys = ['ego', 'mask', 'cm', 'static']
    cons_dict_avg = {k: 0 for k in cons_keys}

  loss_dict_avg = {k: 0 for k in keys }

  for data in tqdm(dataloader):
    loss_dict, output_dict = step(args, data, model, loss, augmentations, optimizer)

    # calculate gradients and then do Adam step
    optimizer.zero_grad()
    total_loss = loss_dict['total_loss']
    total_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    for key in keys:
      loss_dict_avg[key] += loss_dict[key]
    if args.train_consistency:
      for cons_key in cons_keys:
        cons_dict_avg[cons_key] += loss_dict['cons_dict'][cons_key]

  n = len(dataloader)
  for key in keys:
    loss_dict_avg[key] /= n

  if args.train_consistency:
    for cons_key in cons_keys:
      cons_dict_avg[cons_key] /= n
  
    loss_dict_avg['cons_dict'] = cons_dict_avg
  
  return loss_dict_avg, output_dict, data


def eval(args, model, loss, dataloader, augmentations):
  val_loss_sum = 0.

  for data_dict in tqdm(dataloader):
    with torch.no_grad():
      # Get input and target tensor keys
      input_keys = list(filter(lambda x: "input" in x, data_dict.keys()))
      target_keys = list(filter(lambda x: "target" in x, data_dict.keys()))
      tensor_keys = input_keys + target_keys

      # Possibly transfer to Cuda
      if args.cuda:
        for k, v in data_dict.items():
          if k in tensor_keys:
            data_dict[k] = v.cuda(non_blocking=True)
      data_dict = augmentations(data_dict)
      output_dict = model(data_dict)
      loss_dict = loss(output_dict, data_dict)
      val_loss_sum += loss_dict['total_loss']

  val_loss_avg = val_loss_sum / len(dataloader)
  return val_loss_avg

def visualize_output(args, input_dict, output_dict, epoch, writer=None):

  img_l1 = input_dict['input_l1_aug']
  img_l2 = input_dict['input_l2_aug']
  img_r1 = input_dict['input_r1_aug']
  sf_f_s = output_dict['flow_f'][0]
  # sf_f_s = output_dict['flow_f_s'][0]
  # sf_f_d = output_dict['flow_f_d'][0]
  disp_l1_s = output_dict['disp_l1'][0]
  # disp_l1_s = output_dict['disp_l1_s'][0]
  # disp_l1_d = output_dict['disp_l1_d'][0]
  k_l1 = input_dict['input_k_l1_aug']
  k_r1 = input_dict['input_k_r1_aug']
  if args.train_consistency:
    cms_f = output_dict['cms_f'][0]
    img_l1_cms_warp = flow_warp(img_l1, pose2flow(disp2depth_kitti(disp_l1_s, k_l1[:, 0, 0]).squeeze(0), cms_f, k_l1, torch.inverse(k_l1)))
    rigidity_mask = output_dict['rigidity_f'][0]

  flow_f_s = projectSceneFlow2Flow(k_l1, sf_f_s, disp_l1_s)
  # flow_f_d = projectSceneFlow2Flow(k_l1, sf_f_d, disp_l1_d)
  img_l1_warp_s = flow_warp(img_l1, flow_f_s)
  # img_l1_warp_d = flow_warp(img_l1, flow_f_d)
  img_r1_warp_s = _generate_image_left(img_r1, disp_l1_s)
  # img_r1_warp_d = _generate_image_left(img_r1, disp_l1_d)

  if writer is not None:
    writer.add_image('input_l1', img_l1.squeeze(), epoch)
    writer.add_image('input_l2', img_l2.squeeze(), epoch)
    writer.add_image('input_r1', img_r1.squeeze(), epoch)
    writer.add_image('output_l2_s', img_l1_warp_s.squeeze(), epoch)
    # writer.add_image('output_l2_d', img_l1_warp_d.squeeze(), epoch)
    writer.add_image('output_l1_s', img_r1_warp_s.squeeze(), epoch)
    # writer.add_image('output_l1_d', img_r1_warp_d.squeeze(), epoch)
    if args.train_consistency:
      writer.add_image('input_l1', img_l1.squeeze(), epoch)
      writer.add_image('input_l2', img_l2.squeeze(), epoch)
      writer.add_image('input_r1', img_r1.squeeze(), epoch)
      writer.add_image('output_l2_s', img_l1_warp_s.squeeze(), epoch)
      writer.add_image('output_l2_d', img_l1_warp_d.squeeze(), epoch)
      writer.add_image('output_l1_s', img_r1_warp_s.squeeze(), epoch)
      writer.add_image('output_l1_d', img_r1_warp_d.squeeze(), epoch)
      writer.add_image('output_l1_cms_warp', img_l1_cms_warp.squeeze(), epoch)
      writer.add_image('rigidity_mask', rigidity_mask.squeeze(0), epoch)

  return
  

if __name__ == '__main__':
  main()
