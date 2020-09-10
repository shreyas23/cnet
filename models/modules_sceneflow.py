from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging
from .common import conv_bn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride, downsample, pad, dilation, use_bn=True):
      super(BasicBlock, self).__init__()

      self.conv1 = conv_bn(in_ch, out_ch, 3, stride, pad, dilation, use_bn=use_bn)

      if use_bn:
        self.conv2 = nn.Sequential(
          nn.Conv2d(out_ch, out_ch, 3, 1, pad, dilation),
          nn.BatchNorm2d(out_ch))
      else:
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, pad, dilation),

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
    def __init__(self, in_chs, use_bn=True, with_ppm=False):
        super(ResNetEncoder, self).__init__()

        self.in_chs = in_chs

        self.conv1 = nn.Sequential(
          conv_bn(3,  32, 3, 2, 1, 1, use_bn=use_bn),
          conv_bn(32, 32, 3, 1, 1, 1, use_bn=use_bn),
          conv_bn(32, 32, 3, 1, 1, 1, use_bn=use_bn))

        self.res1 = self._make_layer(BasicBlock, 32, 3, 2, 1, 1, use_bn=use_bn)
        self.res2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1, use_bn=use_bn)
        self.res3 = self._make_layer(BasicBlock, 128, 3, 2, 1, 1, use_bn=use_bn)
        self.res4 = self._make_layer(BasicBlock, 256, 3, 2, 1, 1, use_bn=use_bn)

        if with_ppm:
          self.ppm = PPM(
            [32, 32, 64, 128, 128],
            ppm_last_conv_chs=128,
            ppm_inter_conv_chs=128,
            bn_type=bn_type)
        else:
          self.ppm = None

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
      out2 = self.res1(out1)
      out3 = self.res2(out2)
      out4 = self.res3(out3)
      out5 = self.res4(out4)

      if self.ppm is not None:
        out5_2 = self.ppm(out5)
      else:
        out5_2 = None

      return [out1, out2, out3, out4, out5, out5_2]
