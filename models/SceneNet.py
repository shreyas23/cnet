import torch 
import torch.nn as nn
import torch.nn.functional as tf

from sys import exit
from .encoders import *
from .decoders import *
from .modules_sceneflow import interpolate2d_as
from .correlation_package.correlation import Correlation
from .modules_sceneflow import WarpingLayer_SF, upconv, upsample_outputs_as, initialize_msra
    
class SceneNet(nn.Module):
  def __init__(self, args):
    super(SceneNet, self).__init__()
    
    self.search_range = 4
    self.output_level = 4
    
    # encoder
    if args.encoder_name == 'pwc':
      self.num_levels = 7
      self.encoder_chs = [3, 32, 64, 96, 128, 192, 256]
      self.encoder = PWCEncoder(self.encoder_chs)
    elif args.encoder_name == 'resnet':
      self.num_levels = 5
      self.encoder_chs = [32, 32, 64, 128, 256]
      self.encoder = ResNetEncoder(3, self.encoder_chs)
    else:
      raise NotImplementedError

    self.warping_layer_sf = WarpingLayer_SF()

    # decoders
    self.sf_disp_decoders = nn.ModuleList()
    self.pose_decoders = nn.ModuleList()
    # self.sf_decoders = nn.ModuleList()
    # self.disp_decoders = nn.ModuleList()
    # self.mask_decoders = nn.ModuleList()

    self.upconv_layers = nn.ModuleList()

    self.dim_corr = (self.search_range * 2 + 1) ** 2

    for l, ch in enumerate(self.encoder_chs[::-1]):
      if l > self.output_level:
        break

      if l == 0:
        sf_in_ch = self.dim_corr + self.dim_corr + ch + ch
        pose_in_ch = self.dim_corr + ch + ch
      else:
        sf_in_ch = self.dim_corr + self.dim_corr + ch + ch + 32 + 3 + 1
        pose_in_ch = self.dim_corr + ch + ch + 6
        self.upconv_layers.append(upconv(32, 32, 3, 2))

      sf_decoder = FlowDispDecoder(sf_in_ch)
      pose_decoder = PoseDecoder(pose_in_ch)

      self.sf_disp_decoders.append(sf_decoder)
      self.pose_decoders.append(pose_decoder)

    self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}
    self.context_networks = ContextNetwork(32 + 3 + 1)
    self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

    self.init_weights()

    return
  

  def init_weights(self):
    initialize_msra(self.modules())
    return


  def pwc_forward(self, data_dict, img_l1, img_l2, img_r1, img_r2, k1, k2):

    # img_l1 = data_dict['input_l1_aug']
    # img_l2 = data_dict['input_l2_aug']
    # img_r1 = data_dict['input_r1_aug']
    # img_r2 = data_dict['input_r2_aug']
    # k1 = data_dict['input_k_l1']
    # k2 = data_dict['input_k_l2']

    output_dict = {}

    # on the bottom level are original images
    l1_pyramid = self.encoder(img_l1) + [img_l1]
    l2_pyramid = self.encoder(img_l2) + [img_l2]
    r1_pyramid = self.encoder(img_r1) + [img_r1]
    r2_pyramid = self.encoder(img_r2) + [img_r2]

    # outputs
    sceneflows_f = []
    sceneflows_b = []
    poses_f = []
    poses_b = []
    disps_1 = []
    disps_2 = []

    for l, (l1, l2, r1, r2) in enumerate(zip(l1_pyramid, l2_pyramid, r1_pyramid, r2_pyramid)):

      # warping
      if l == 0:
        l2_warp = l2
        l1_warp = l1
      else:
        flow_f = interpolate2d_as(flow_f, l1, mode="bilinear")
        flow_b = interpolate2d_as(flow_b, l1, mode="bilinear")
        disp_l1 = interpolate2d_as(disp_l1, l1, mode="bilinear")
        disp_l2 = interpolate2d_as(disp_l2, l1, mode="bilinear")
        pose_f_out = interpolate2d_as(pose_f_out, l1, mode="bilinear")
        pose_b_out = interpolate2d_as(pose_b_out, l1, mode="bilinear")

        l1_out = self.upconv_layers[l-1](l1_out)
        l2_out = self.upconv_layers[l-1](l2_out)
        l2_warp = self.warping_layer_sf(l2, flow_f, disp_l1, k1, data_dict['aug_size'])  # becuase K can be changing when doing augmentation
        l1_warp = self.warping_layer_sf(l1, flow_b, disp_l2, k2, data_dict['aug_size'])

      # correlation
      out_corr_f = Correlation.apply(l1, l2_warp, self.corr_params)
      out_corr_b = Correlation.apply(l2, l1_warp, self.corr_params)
      out_corr_relu_f = self.leakyRELU(out_corr_f)
      out_corr_relu_b = self.leakyRELU(out_corr_b)

      disp_corr_l1 = Correlation.apply(l1, r1, self.corr_params)
      disp_corr_l2 = Correlation.apply(l2, r2, self.corr_params)
      disp_corr_l1 = self.leakyRELU(disp_corr_l1)
      disp_corr_l2 = self.leakyRELU(disp_corr_l2)

      # monosf estimator
      if l == 0:
        l1_out, flow_f, disp_l1 = self.sf_disp_decoders[l](torch.cat([out_corr_relu_f, disp_corr_l1, l1, l2], dim=1))
        l2_out, flow_b, disp_l2 = self.sf_disp_decoders[l](torch.cat([out_corr_relu_b, disp_corr_l2, l2, l1], dim=1))
        pose_f_out, pose_f = self.pose_decoders[l](torch.cat([out_corr_relu_f, l1, l2], dim=1))
        pose_b_out, pose_b = self.pose_decoders[l](torch.cat([out_corr_relu_b, l2, l1], dim=1))
      else:
        l1_out, flow_f_res, disp_l1 = self.sf_disp_decoders[l](torch.cat([out_corr_relu_f, disp_corr_l1, l1, l2, l1_out, flow_f, disp_l1], dim=1))
        l2_out, flow_b_res, disp_l2 = self.sf_disp_decoders[l](torch.cat([out_corr_relu_b, disp_corr_l2, l2, l1, l2_out, flow_b, disp_l2], dim=1))
        flow_f = flow_f + flow_f_res
        flow_b = flow_b + flow_b_res

        pose_f_out, pose_f = self.pose_decoders[l](torch.cat([out_corr_relu_f, l1, l2, pose_f_out], dim=1))
        pose_b_out, pose_b = self.pose_decoders[l](torch.cat([out_corr_relu_b, l2, l1, pose_b_out], dim=1))

      # upsampling or post-processing
      if l != self.output_level:
        disp_l1 = torch.sigmoid(disp_l1)  # * 0.3
        disp_l2 = torch.sigmoid(disp_l2)  # * 0.3
        sceneflows_f.append(flow_f)
        sceneflows_b.append(flow_b)                
        poses_f.append(pose_f)
        poses_b.append(pose_b)
        disps_1.append(disp_l1)
        disps_2.append(disp_l2)
      else:
        flow_res_f, disp_l1 = self.context_networks(torch.cat([l1_out, flow_f, disp_l1], dim=1))
        flow_res_b, disp_l2 = self.context_networks(torch.cat([l2_out, flow_b, disp_l2], dim=1))
        flow_f = flow_f + flow_res_f
        flow_b = flow_b + flow_res_b
        sceneflows_f.append(flow_f)
        sceneflows_b.append(flow_b)
        poses_f.append(pose_f)
        poses_b.append(pose_b)
        disps_1.append(disp_l1)
        disps_2.append(disp_l2)                
        break

    l1_rev = l1_pyramid[::-1]

    output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], l1_rev)
    output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], l1_rev)
    output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], l1_rev)
    output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], l1_rev)
    output_dict['poses_f'] = poses_f[::-1]
    output_dict['poses_b'] = poses_b[::-1]

    return output_dict

  def forward(self, input_dict):
    output_dict = self.pwc_forward(input_dict, 
                                   input_dict['input_l1_aug'], 
                                   input_dict['input_l2_aug'], 
                                   input_dict['input_r1_aug'], 
                                   input_dict['input_r2_aug'],
                                   input_dict['input_k_l1_aug'],
                                   input_dict['input_k_l2_aug'])

    # flip inputs for extra training dataset
    if self.training:
      input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
      input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
      input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
      input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
      k_r1_flip = input_dict["input_k_r1_flip_aug"]
      k_r2_flip = input_dict["input_k_r2_flip_aug"]

      output_dict_r = self.pwc_forward(input_dict,
                                       input_r1_flip,
                                       input_r2_flip,
                                       input_l1_flip,
                                       input_l2_flip,
                                       k_r1_flip,
                                       k_r2_flip)

      output_dict['output_dict_r'] = output_dict_r

    return output_dict