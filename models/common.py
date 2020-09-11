import torch
import torch.nn as nn
import torch.nn.functional as tf

""" some functions borrowed and modified from SENSE
    https://github.com/NVlabs/SENSE/blob/master/sense/models/common.py
"""


def conv(in_chs, out_chs, kernel_size=3, stride=1, dilation=1, bias=True, use_relu=True, use_bn=True):
  layers = []
  layers.append(nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size - 1) * dilation) // 2, bias=bias))
  if use_bn:
    layers.append(nn.BatchNorm2d(out_chs))
  if use_relu:
    layers.append(nn.LeakyReLU(0.1, inplace=True))
  
  return nn.Sequential(*layers)


def apply_rigidity_mask(static, dynamic, rigidity_mask, mask_thresh=0.5, use_thresh=True):
  _, ch, _, _ = static.shape
  if use_thresh:
    rigidity_mask = 1. * (rigidity_mask >= mask_thresh).repeat(1, ch, 1, 1).float()
  merged_flow = static * (1. - rigidity_mask) + dynamic * rigidity_mask
  return merged_flow


def flow_warp(x, flo):
    """warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    mask[mask.data < 0.9999] = 0
    mask[mask.data > 0] = 1
    
    return output*mask

def disp_warp(rim, disp):
    """
    warp stereo image (right image) with disparity
    rim: [B, C, H, W] image/tensor
    disp: [B, 1, H, W] (left) disparity
    for disparity (left), we have
        left_image(x,y) = right_image(x-d(x,y),y)
    """
    B, C, H, W = rim.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if rim.is_cuda:
        grid = grid.cuda()
    vgrid = grid

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*(vgrid[:,0,:,:]-disp[:,0,:,:])/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    return nn.functional.grid_sample(rim, vgrid.permute(0,2,3,1))