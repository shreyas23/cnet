from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging

from .common import conv


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride, downsample, pad, dilation, use_bn=True):
      super(BasicBlock, self).__init__()

      self.conv1 = conv(in_ch, out_ch, 3, stride, pad, dilation, use_bn=use_bn)
      self.conv2 = conv(out_ch, out_ch, 3, 1, pad, dilation, use_relu=False, use_bn=use_bn)

      self.downsample = downsample
      self.stride = stride

    def forward(self, x):
      out = self.conv1(x)
      out = self.conv2(out)

      if self.downsample is not None:
          x = self.downsample(x)

      out += x

      return out


class ResNetEncoder(nn.Module):
    def __init__(self, in_chs, conv_chs=None, with_ppm=False, use_bn=True):
        super(ResNetEncoder, self).__init__()
        if conv_chs is None:
          self.conv_chs = [32, 32, 64, 128, 128]
        else:
          self.conv_chs = conv_chs

        self.in_chs = self.conv_chs[0]

        self.conv1 = nn.Sequential(
          conv(in_chs, self.in_chs, 3, 2, 1, 1, use_bn=use_bn),
          conv(self.in_chs, self.in_chs, 3, 1, 1, 1, use_bn=use_bn),
          conv(self.in_chs, self.in_chs, 3, 1, 1, 1, use_bn=use_bn))

        self.res_layers = nn.ModuleList()
        for conv_ch in self.conv_chs[1:]:
            self.res_layers.append(self._make_layer(BasicBlock, conv_ch, 3, 2, 1, 1, use_bn=use_bn))

        # if with_ppm:
        #   self.ppm = PPM(
        #     [32, 32, 64, 128, 128],
        #     ppm_last_conv_chs=128,
        #     ppm_inter_conv_chs=128,
        #     bn_type=bn_type)
        # else:
        #   self.ppm = None

    def _make_layer(self, block, chs, blocks, stride, pad, dilation, use_bn=True):
      downsample = None
      if stride != 1 or self.in_chs != chs * block.expansion:
        if use_bn:
          downsample = nn.Sequential(
            nn.Conv2d(self.in_chs, chs * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(chs * block.expansion))
        else:
          downsample = nn.Conv2d(self.in_chs, chs * block.expansion, kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(block(self.in_chs, chs, stride, downsample, pad, dilation, use_bn=use_bn))
        self.in_chs = chs * block.expansion
        for i in range(1, blocks):
          layers.append(block(self.in_chs, chs, 1, None, pad, dilation, use_bn=use_bn))

        return nn.Sequential(*layers)

    def forward(self, x):
      out1 = self.conv1(x)
      outs = [out1]

      for res_layer in self.res_layers:
        outs.append(res_layer(outs[-1]))
      # if self.ppm is not None:
      #   outs.append(self.ppm(outs[-1]))
      # else:
      #   outs.append(None)

      return outs


class PWCEncoder(nn.Module):
  def __init__(self, conv_chs=None):
    super(PWCEncoder, self).__init__()
      
    if conv_chs is None:
      self.conv_chs = [3, 32, 64, 128, 256, 512]
    else:
      self.conv_chs = conv_chs
          
    self.convs = nn.ModuleList()
      
    for in_ch, out_ch in zip(self.conv_chs[:-1], self.conv_chs[1:]):
      layers = nn.Sequential(
        conv(in_ch, out_ch, stride=2),
        conv(out_ch, out_ch))
      self.convs.append(layers)
          
  def forward(self, x):
    fp = []
    for conv in self.convs:
      fp.append(conv(x))
    return fp
