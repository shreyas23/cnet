import os
import argparse
import datetime
import numpy as np
from sys import exit
from time import time
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from augmentations import Augmentation_SceneFlow, Augmentation_Resize_Only

from datasets.kitti_raw_monosf import CarlaDataset, KITTI_Raw_KittiSplit_Train, KITTI_Raw_KittiSplit_Valid

from models.SceneNet import SceneNet
from models.model_monosceneflow import MonoSceneFlow

from utils.inverse_warp import flow_warp, pose2flow, inverse_warp
from utils.sceneflow_util import projectSceneFlow2Flow, disp2depth_kitti

from losses import Loss_SceneFlow_SelfSup
from losses_consistency import Loss_SceneFlow_SelfSup_Pose, _generate_image_left, _adaptive_disocc_detection


parser = argparse.ArgumentParser(description="Self Supervised Joint Learning of Scene Flow, Disparity, Rigid Camera Motion, and Motion Segmentation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# runtime params
parser.add_argument('--data_root', help='path to dataset', required=True)
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to run')
parser.add_argument('--start_epoch', type=int, default=0, help='resume from checkpoint (using experiment name)')
parser.add_argument('--cuda', type=bool, default=True, help='use gpu?')
parser.add_argument('--no_logging', type=bool, default=False, help="are you logging this experiment?")
parser.add_argument('--log_dir', type=str, default="/external/cnet/checkpoints", help="are you logging this experiment?")
parser.add_argument('--exp_name', type=str, default='test', help='name of experiment, chkpts stored in checkpoints/experiment')
parser.add_argument('--validate', type=bool, default=False, help='set to true if validating model')
parser.add_argument('--ckpt', type=str, default="", help="path to model checkpoint if using one")

# module params
parser.add_argument('--model_name', type=str, default="scenenet", help="name of model")
parser.add_argument('--encoder_name', type=str, default="resnet", help="which encoder to use for model")

# dataset params
parser.add_argument('--dataset_name', default='KITTI', help='KITTI or Carla')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_views', type=int, default=2, help="number of views present in training data")
parser.add_argument('--num_examples', type=int, default=-1, help="number of examples to train on per epoch")
parser.add_argument('--num_workers', type=int, default=8, help="number of workers for the dataloader")
parser.add_argument('--shuffle_dataset', type=bool, default=True, help='shuffle the dataset?')
parser.add_argument('--resize_only', type=bool, default=True, help='only do resize augmentation on input data')

# learning params
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--lr_sched_type', type=str, default="none", help="path to model checkpoint if using one")
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
parser.add_argument('--multi_gpu', type=bool, default=False, help='use multiple gpus')
parser.add_argument('--debugging', type=bool, default=False, help='are you debugging?')
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

    if args.model_name == 'scenenet':
      model = SceneNet(args)
      loss = Loss_SceneFlow_SelfSup_Pose(args)
    elif args.model_name == 'monoflow':
      model = MonoSceneFlow(args)
      loss = Loss_SceneFlow_SelfSup(args)
    else:
      raise NotImplementedError

    # define dataset
    train_dataset = KITTI_Raw_KittiSplit_Train(args, DATA_ROOT, num_examples=args.num_examples, flip_augmentations=False)

    if args.validate:
      val_num_examples = int(args.num_examples / 5) if args.num_examples > 0 else -1
      val_dataset = KITTI_Raw_KittiSplit_Valid(args, DATA_ROOT, num_examples=val_num_examples)
    else:
      val_dataset = None

    # define augmentations
    if args.resize_only:
      print("Only resizing")
      augmentations = Augmentation_Resize_Only(args)
    else:
      augmentations = Augmentation_SceneFlow(args)
  else:
    raise NotImplementedError

  train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=args.shuffle_dataset, num_workers=args.num_workers, pin_memory=True)
  val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=8, pin_memory=True) if val_dataset else None

  # load the model
  print("Loding model and augmentations and placing on gpu...")

  if args.cuda:
    augmentations = augmentations.cuda()
    model = model.cuda()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"The model has {num_params} parameters")

  # load optimizer and lr scheduler
  optimizer = Adam(model.parameters(), lr=args.lr, betas=[args.momentum, args.beta], weight_decay=args.weight_decay)

  if args.lr_sched_type == 'plateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, verbose=True, mode='min', patience=10)
  elif args.lr_sched_type == 'step':
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 100, 150])
  elif args.lr_sched_type == 'none':
    lr_scheduler=None

  # set up logging
  if not args.no_logging:
    log_dir = os.path.join(args.log_dir, args.exp_name)
    if not os.path.isdir(log_dir):
      os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%H%M%S-%Y%m%d"))
    writer = SummaryWriter(log_dir)

  if args.ckpt != "":
    state_dict = torch.load(args.ckpt)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
      name = k[7:] 
      new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
  elif args.start_epoch > 0:
    load_epoch = args.start_epoch - 1
    ckpt_fp = os.path.join(log_dir, f"{load_epoch}.ckpt")

    print(f"Loading model from {ckpt_fp}...")

    ckpt = torch.load(ckpt_fp)
    assert (ckpt['epoch'] == load_epoch), "epoch from state dict does not match with args"
    model.load_state_dict(ckpt)

  # run training loop
  for epoch in range(args.start_epoch, args.epochs + 1):
    print(f"Training epoch: {epoch}...")
    train_loss_avg_dict, output_dict, input_dict = train_one_epoch(args, model, loss, train_dataloader, optimizer, augmentations, lr_scheduler)
    print(f"\t Epoch {epoch} train loss avg:")
    pprint(train_loss_avg_dict)

    if val_dataset is not None:
      print(f"Validation epoch: {epoch}...")
      val_loss_avg = eval(args, model, loss, val_dataloader, augmentations)
      print(f"\t Epoch {epoch} val loss avg: {val_loss_avg}")

    if not args.no_logging:
      writer.add_scalar('loss/train', train_loss_avg_dict['total_loss'], epoch)
      # writer.add_scalar('loss/val', val_loss_avg, epoch)
      writer.add_scalar('loss/train/sf', train_loss_avg_dict['sf'], epoch)
      writer.add_scalar('loss/train/dp', train_loss_avg_dict['dp'], epoch)
      writer.add_scalar('loss/train/po', train_loss_avg_dict['po'], epoch)
      writer.add_scalar('loss/train/s_2', train_loss_avg_dict['s_2'], epoch)
      writer.add_scalar('loss/train/s_3', train_loss_avg_dict['s_3'], epoch)
      writer.add_scalar('loss/train/s_3s', train_loss_avg_dict['s_3s'], epoch)

      if epoch % 20 == 0:
        visualize_output(args, input_dict, output_dict, epoch, writer)

      # if args.train_consistency:
      #   writer.add_scalar('loss/train/cons/total', train_loss_avg_dict['cons'], epoch)
      #   writer.add_scalar('loss/train/cons/cm', train_loss_avg_dict['cons_dict']['cm'], epoch)
      #   writer.add_scalar('loss/train/cons/mask', train_loss_avg_dict['cons_dict']['mask'], epoch)
      #   writer.add_scalar('loss/train/cons/static', train_loss_avg_dict['cons_dict']['static'], epoch)
      #   writer.add_scalar('loss/train/cons/dynamic', train_loss_avg_dict['cons_dict']['dynamic'], epoch)
      #   writer.add_scalar('loss/train/cons/ego', train_loss_avg_dict['cons_dict']['ego'], epoch)

    assert (not torch.isnan(train_loss_avg_dict['total_loss'])), "avg training loss is nan"

    if args.lr_sched_type == 'plateau':
      lr_scheduler.step(train_loss_avg_dict['total_loss'])
    elif args.lr_sched_type == 'step':
      lr_scheduler.step(epoch)

    # save model
    if not args.no_logging:
      if epoch % 1000 == 0 or epoch == args.epochs:
        fp = os.path.join(log_dir, f"{epoch}.ckpt")

        torch.save(model.state_dict(), fp)

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
  if args.model_name == 'scenenet':
    keys.append('po')

  if args.train_consistency:
    keys.append('cons')
    cons_keys = ['ego', 'mask', 'cm', 'static']
    cons_dict_avg = {k: 0 for k in cons_keys}

  loss_dict_avg = {k: 0 for k in keys }

  for data in tqdm(dataloader):
    loss_dict, output_dict = step(args, data, model, loss, augmentations, optimizer)
    print(output_dict['flow_f'])
    print(output_dict['disp_l1'])
    exit(0)

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


def visualize_output(args, input_dict, output_dict, epoch, writer):

  assert (writer is not None), "tensorboard writer not provided"

  img_l1 = input_dict['input_l1_aug']
  img_l2 = input_dict['input_l2_aug']
  img_r1 = input_dict['input_r1_aug']
  sf_f = output_dict['flow_f'][0]    
  sf_b = output_dict['flow_b'][0]    
  disp_l1 = output_dict['disp_l1'][0]
  disp_l2 = output_dict['disp_l2'][0]
  k_l1 = input_dict['input_k_l1_aug']
  k_r1 = input_dict['input_k_r1_aug']
  k_l2 = input_dict['input_k_l2_aug']
  k_r2 = input_dict['input_k_r2_aug']
  baseline = input_dict['baseline']
  aug_size = input_dict['aug_size']
  if args.model_name == 'scenenet':
    pose_f = output_dict['poses_f'][0]
    pose_b = output_dict['poses_b'][0]

  # un-normalize disparity
  _, _, h_dp, w_dp = sf_f.size()
  disp_l1 = disp_l1 * w_dp
  disp_l2 = disp_l2 * w_dp

  # warp img_r1 
  img_r1_warp = _generate_image_left(img_r1, disp_l1)

  if args.model_name == 'scenenet':
    # inverse warp img_l2 through pose_f
    depth_l1 = disp2depth_kitti(disp_l1, k_l1[:, 0, 0], baseline)
    depth_l2 = disp2depth_kitti(disp_l2, k_l2[:, 0, 0], baseline)
    cam_flow_f = pose2flow(depth_l1, pose_f, k_l1, torch.inverse(k_l1))
    cam_flow_b = pose2flow(depth_l2, pose_b, k_l2, torch.inverse(k_l2))
    cam_occ_f = _adaptive_disocc_detection(cam_flow_b)
    img_l2_cam_warp = flow_warp(img_l2, cam_flow_f)

  # scale
  local_scale = torch.zeros_like(aug_size)
  local_scale[:, 0] = h_dp
  local_scale[:, 1] = w_dp

  # inverse warp img_l1 through sf_f
  pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
  _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
  img_l2_warp = reconstructImg(coord1, img_l2_aug)

  flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
  occ_f = _adaptive_disocc_detection(flow_b).detach()

  if writer is not None:
    writer.add_image('input_l1', img_l1.squeeze(), epoch)
    writer.add_image('input_l2', img_l2.squeeze(), epoch)
    writer.add_image('input_r1', img_r1.squeeze(), epoch)
    writer.add_image('img_l2_warp', img_l2_warp.squeeze(), epoch)
    writer.add_image('img_r1_warp', img_r1_warp.squeeze(), epoch)
    writer.add_image('occ_f', occ_f.squeeze(), epoch)
    if args.model_name == 'scenenet':
      writer.add_image('img_l2_cam_warp', img_l2_cam_warp.squeeze(), epoch)
      writer.add_image('cam_occ_f', cam_occ_f.squeeze(), epoch)

  return 

if __name__ == '__main__':
  main()
