from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms


def get_grid(x):
    grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1,
                                                       1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1,
                                                       x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([grid_H, grid_V], 1)
    grids_cuda = grid.float().requires_grad_(False).cuda()
    return grids_cuda


class WarpingLayer_Flow(nn.Module):
    def __init__(self):
        super(WarpingLayer_Flow, self).__init__()

    def forward(self, x, flow):
        flo_list = []
        flo_w = flow[:, 0] * 2 / max(x.size(3) - 1, 1)
        flo_h = flow[:, 1] * 2 / max(x.size(2) - 1, 1)
        flo_list.append(flo_w)
        flo_list.append(flo_h)
        flow_for_grid = torch.stack(flo_list).transpose(0, 1)
        grid = torch.add(get_grid(x), flow_for_grid).transpose(
            1, 2).transpose(2, 3)
        x_warp = tf.grid_sample(x, grid)

        mask = torch.ones(x.size(), requires_grad=False).cuda()
        mask = tf.grid_sample(mask, grid)
        mask = (mask >= 1.0).float()

        return x_warp * mask


class WarpingLayer_SF(nn.Module):
    def __init__(self):
        super(WarpingLayer_SF, self).__init__()

    def forward(self, x, sceneflow, disp, k1, input_size):

        _, _, h_x, w_x = x.size()
        disp = interpolate2d_as(disp, x) * w_x

        local_scale = torch.zeros_like(input_size)
        local_scale[:, 0] = h_x
        local_scale[:, 1] = w_x

        pts1, k1_scale = pixel2pts_ms(k1, disp, local_scale / input_size)
        _, _, coord1 = pts2pixel_ms(k1_scale, pts1, sceneflow, [h_x, w_x])

        grid = coord1.transpose(1, 2).transpose(2, 3)
        x_warp = tf.grid_sample(x, grid)

        mask = torch.ones_like(x, requires_grad=False)
        mask = tf.grid_sample(mask, grid)
        mask = (mask >= 1.0).float()

        return x_warp * mask


def initialize_msra(modules):
    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.LeakyReLU):
            pass

        elif isinstance(layer, nn.Sequential):
            pass


def upsample_outputs_as(input_list, ref_list):
    output_list = []
    for ii in range(0, len(input_list)):
        output_list.append(interpolate2d_as(input_list[ii], ref_list[ii]))

    return output_list


def conv(in_chs, out_chs, kernel_size=3, stride=1, dilation=1, bias=True, use_relu=True, use_bn=True):
  layers = []
  layers.append(nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size - 1) * dilation) // 2, bias=bias))
  if use_bn:
    layers.append(nn.BatchNorm2d(out_chs))
  if use_relu:
    layers.append(nn.LeakyReLU(0.1, inplace=True))
  
  return nn.Sequential(*layers)


class upconv_interpolate(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv_interpolate, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(
            x, scale_factor=self.scale, mode='nearest')
        return self.conv1(x)


def upconv_transpose(in_ch, out_ch, kernel_size=4, stride=2, padding=1, relu=True):
    if relu:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding)


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
    disp1 = self.conv_d1(x_out)  # * 0.3

    return sf, disp1