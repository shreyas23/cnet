import torch
import torch.nn as nn
import torch.nn.functional as tf

from .common import conv


class PoseDecoder(nn.Module):
  def __init__(self, in_ch, num_refs=1, use_bn=True):
    super(PoseDecoder, self).__init__()

    conv_chs = [64, 128, 128, 256, 256, 6*num_refs]
    self.convs = nn.Sequential(
      conv(in_ch, in_ch, kernel_size=7, use_bn=use_bn),
      conv(in_ch, conv_chs[0], kernel_size=5, use_bn=use_bn),
      conv(conv_chs[0], conv_chs[1], stride=1, use_bn=use_bn),
      conv(conv_chs[1], conv_chs[2], stride=1, use_bn=use_bn),
      conv(conv_chs[2], conv_chs[3], stride=1, use_bn=use_bn),
      conv(conv_chs[3], conv_chs[4], stride=1, use_bn=use_bn),
      conv(conv_chs[4], conv_chs[5], stride=1, use_relu=False, use_bn=False))

  def forward(self, x):
    x = self.convs(x)
    motion_pred = x.mean(3).mean(2) * 0.01
    return x, motion_pred


class MaskNetDecoder(nn.Module):
  def __init__(self, in_ch, num_refs=1, use_bn=True):
    super(MaskNetDecoder, self).__init__()

    conv_chs = [256, 256, 128, 64, 32, 16]
    self.convs = nn.Sequential(
      conv(in_ch, conv_chs[0], kernel_size=7, stride=1, use_bn=use_bn),
      conv(conv_chs[0], conv_chs[1], kernel_size=5, stride=1, use_bn=use_bn),
      conv(conv_chs[1], conv_chs[2], stride=1, use_bn=use_bn),
      conv(conv_chs[2], conv_chs[3], stride=1, use_bn=use_bn),
      conv(conv_chs[3], conv_chs[4], stride=1, use_bn=use_bn),
      conv(conv_chs[4], conv_chs[5], stride=1, use_bn=use_bn),
      conv(conv_chs[-1], num_refs, use_relu=False, use_bn=False),
      torch.nn.Sigmoid())

    def forward(self, x):
      x = self.convs(x)
      return x


class PoseMaskNet(nn.Module):
  def __init__(self, in_ch, num_refs=1, use_bn=True):
    super(PoseMaskNet, self).__init__()
    shared_chs = [128, 256]
    cam_chs = [256, 6*num_refs]
    mask_chs = [256, 128, 64, 32, 16, num_refs]

    self.shared_convs = nn.Sequential(
      conv(in_ch, shared_chs[0], kernel_size=7, stride=1, use_bn=use_bn),
      conv(shared_chs[0], shared_chs[1], kernel_size=5, stride=1, use_bn=use_bn))

    self.cam_convs = nn.Sequential(
      conv(shared_chs[-1], cam_chs[0], stride=1, kernel_size=3, use_bn=use_bn),
      conv(cam_chs[0], cam_chs[1], stride=1, kernel_size=3, use_relu=False, use_bn=False))

    self.mask_convs = nn.Sequential(
      conv(shared_chs[-1], mask_chs[0], stride=1, kernel_size=3, use_bn=use_bn),
      conv(mask_chs[0], mask_chs[1], stride=1, kernel_size=3, use_bn=use_bn),
      conv(mask_chs[1], mask_chs[2], stride=1, kernel_size=3, use_bn=use_bn),
      conv(mask_chs[2], mask_chs[3], stride=1, kernel_size=3, use_bn=use_bn),
      conv(mask_chs[3], mask_chs[4], stride=1, kernel_size=5, use_bn=use_bn),
      conv(mask_chs[4], mask_chs[5], stride=1, kernel_size=7, use_relu=False, use_bn=False),
      nn.Sigmoid())

  def forward(self, x):
    shared_out = self.shared_convs(x)
    cam_out = self.cam_convs(shared_out)
    pred_mask = self.mask_convs(shared_out)
    pred_cm = 0.01 * cam_out.mean(3).mean(2)

    return pred_cm, pred_mask


class FlowDispDecoder(nn.Module):
  def __init__(self, ch_in, use_bn=True):
    super(FlowDispDecoder, self).__init__()

    self.convs = nn.Sequential(
      conv(ch_in, 128, use_bn=use_bn),
      conv(128, 128, use_bn=use_bn),
      conv(128, 96, use_bn=use_bn),
      conv(96, 64, use_bn=use_bn),
      conv(64, 32, use_bn=use_bn))

    self.conv_sf = conv(32, 3, use_relu=False, use_bn=False)
    self.conv_d1 = conv(32, 1, use_relu=False, use_bn=False)

  def forward(self, x):
    x_out = self.convs(x)
    sf = self.conv_sf(x_out)
    disp1 = self.conv_d1(x_out)

    return x_out, sf, disp1


class FlowDispSplitDecoder(nn.Module):
  def __init__(self, ch_in, use_bn=True):
    super(FlowDispSplitDecoder, self).__init__()

    self.disp_convs = nn.Sequential(
      conv(ch_in, 128, use_bn=use_bn),
      conv(128, 128, use_bn=use_bn),
      conv(128, 96, use_bn=use_bn),
      conv(96, 64, use_bn=use_bn),
      conv(64, 32, use_bn=use_bn),
      conv(32, 16, use_bn=use_bn))

    self.sf_convs = nn.Sequential(
      conv(ch_in, 128, use_bn=use_bn),
      conv(128, 128, use_bn=use_bn),
      conv(128, 96, use_bn=use_bn),
      conv(96, 64, use_bn=use_bn),
      conv(64, 32, use_bn=use_bn),
      conv(32, 16, use_bn=use_bn))

    self.conv_sf = conv(32, 3, use_relu=False, use_bn=False)
    self.conv_d1 = conv(32, 1, use_relu=False, use_bn=False)

  def forward(self, x):
    disp_out = self.disp_convs(x)
    sf_out = self.sd_convs(x)
    sf = self.conv_sf(sf_out)
    disp1 = self.conv_d1(disp_out)
    x_out = torch.cat([disp_out, sf_out], dim=1)

    return x_out, sf, disp1


class ContextNetwork(nn.Module):
  def __init__(self, ch_in, use_bn=True):
    super(ContextNetwork, self).__init__()

    self.convs = nn.Sequential(
      conv(ch_in, 128, 3, 1, 1, use_bn=use_bn),
      conv(128, 128, 3, 1, 2, use_bn=use_bn),
      conv(128, 128, 3, 1, 4, use_bn=use_bn),
      conv(128, 96, 3, 1, 8, use_bn=use_bn),
      conv(96, 64, 3, 1, 16, use_bn=use_bn),
      conv(64, 32, 3, 1, 1, use_bn=use_bn))

    self.conv_sf = conv(32, 3, use_relu=False, use_bn=False)
    self.conv_d1 = nn.Sequential(
      conv(32, 1, use_relu=False, use_bn=False), 
      torch.nn.Sigmoid())

  def forward(self, x):
    x_out = self.convs(x)
    sf = self.conv_sf(x_out)
    disp1 = self.conv_d1(x_out) * 0.3

    return sf, disp1