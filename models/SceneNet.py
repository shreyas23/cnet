from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging

from models.correlation_package.correlation import Correlation

from .modules_sceneflow import get_grid, WarpingLayer_SF
from .modules_sceneflow import initialize_msra, upsample_outputs_as
from .modules_sceneflow import upconv
from .modules_sceneflow import FeatureExtractor, MonoStaticFlowDecoder, MonoDynamicFlowDecoder, ContextNetwork

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import flow_horizontal_flip, intrinsic_scale, get_pixelgrid, post_processing

class MonoSceneFlow(nn.Module):
    def __init__(self, args):
        super(MonoSceneFlow, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_sf = WarpingLayer_SF()
        
        # self.flow_estimators = nn.ModuleList()
        self.static_flow_estimators = nn.ModuleList()
        self.dynamic_flow_estimators = nn.ModuleList()

        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + 32 + 3 + 1
                self.upconv_layers.append(upconv(32, 32, 3, 2))

            # Add static and dynamic decoders
            static_layer_sf= MonoStaticFlowDecoder(num_ch_in)            
            dynamic_layer_sf = MonoDynamicFlowDecoder(num_ch_in)            

            # self.flow_estimators.append(layer_sf)
            self.static_flow_estimators.append(static_layer_sf)
            self.dynamic_flow_estimators.append(dynamic_layer_sf)

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.static_context_networks = ContextNetwork(32 + 3 + 1)
        self.dynamic_context_networks = ContextNetwork(32 + 3 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2, motion_mask_f, motion_mask_b):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f_s = []
        sceneflows_f_d = []
        sceneflows_b_s = []
        sceneflows_b_d = []
        disps_1_s = []
        disps_1_d = []
        disps_2_s = []
        disps_2_d = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f_s = interpolate2d_as(flow_f_s, x1, mode="bilinear")
                flow_f_d = interpolate2d_as(flow_f_d, x1, mode="bilinear")
                flow_b_s = interpolate2d_as(flow_b_s, x1, mode="bilinear")
                flow_b_d = interpolate2d_as(flow_b_d, x1, mode="bilinear")

                disp_l1_s = interpolate2d_as(disp_l1_s, x1, mode="bilinear")
                disp_l1_d = interpolate2d_as(disp_l1_d, x1, mode="bilinear")
                disp_l2_s = interpolate2d_as(disp_l2_s, x1, mode="bilinear")
                disp_l2_d = interpolate2d_as(disp_l2_d, x1, mode="bilinear")

                # merge static and dynamic masks here to use for warping/correlation???????????????????
                flow_comb_f = flow_f_s * (1 - motion_mask_f) + flow_f_d * motion_mask_f
                flow_comb_b = flow_b_s * (1 - motion_mask_b) + flow_b_d * motion_mask_b

                disp_l1_comb_f = disp_l1_s * (1 - motion_mask_f) + disp_l1_d * motion_mask_f
                disp_l2_comb_b = disp_l2_s * (1 - motion_mask_b) + disp_l2_d * motion_mask_b

                x2_out = x1_out_s * (1 - motion_mask_b) + x1_out_d * motion_mask_b
                x1_out = x2_out_s * (1 - motion_mask_f) + x2_out_d * motion_mask_f

                x1_out = self.upconv_layers[l-1](x1_out)
                x2_out = self.upconv_layers[l-1](x2_out)

                x2_warp = self.warping_layer_sf(x2, flow_comb_f, disp_l1_comb_f, k1, input_dict['aug_size'])  # because K can be changing when doing augmentation
                x1_warp = self.warping_layer_sf(x1, flow_comb_b, disp_l2_comb_b, k2, input_dict['aug_size'])

            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)

            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # monosf estimator
            if l == 0:
                x1_out_s, flow_f_s, disp_l1_s = self.static_flow_estimators[l](torch.cat([out_corr_relu_f, x1], dim=1))
                x1_out_d, flow_f_d, disp_l1_d = self.dynamic_flow_estimators[l](torch.cat([out_corr_relu_f, x1], dim=1))

                x2_out_s, flow_b_s, disp_l2_s = self.static_flow_estimators[l](torch.cat([out_corr_relu_b, x2], dim=1))
                x2_out_d, flow_b_d, disp_l2_d = self.dynamic_flow_estimators[l](torch.cat([out_corr_relu_b, x2], dim=1))
            else:
                x1_out_s, flow_f_res_s, disp_l1_s = self.static_flow_estimators[l](torch.cat([out_corr_relu_f, x1, x1_out_s, flow_f_s, disp_l1_s], dim=1))
                x1_out_d, flow_f_res_d, disp_l1_d = self.dynamic_flow_estimators[l](torch.cat([out_corr_relu_f, x1, x1_out_d, flow_f_d, disp_l1_d], dim=1))

                x2_out_s, flow_b_res_s, disp_l2_s = self.static_flow_estimators[l](torch.cat([out_corr_relu_b, x2, x2_out_s, flow_b_s, disp_l2_s], dim=1))
                x2_out_d, flow_b_res_d, disp_l2_d = self.dynamic_flow_estimators[l](torch.cat([out_corr_relu_b, x2, x2_out_d, flow_b_d, disp_l2_d], dim=1))

                flow_f_s = flow_f_s + flow_f_res_s
                flow_f_d = flow_f_d + flow_f_res_d

                flow_b_s = flow_b_s + flow_b_res_s
                flow_b_d = flow_b_d + flow_b_res_d

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1_s = self.sigmoid(disp_l1_s) * 0.3
                disp_l1_d = self.sigmoid(disp_l1_d) * 0.3
                disp_l2_s = self.sigmoid(disp_l2_s) * 0.3
                disp_l2_d = self.sigmoid(disp_l2_d) * 0.3

                sceneflows_f_s.append(flow_f_s)
                sceneflows_f_d.append(flow_f_d)
                sceneflows_b_s.append(flow_b_s)
                sceneflows_b_d.append(flow_b_d)

                disps_1_s.append(disp_l1_s)
                disps_1_d.append(disp_l1_d)
                disps_2_s.append(disp_l2_s)
                disps_2_d.append(disp_l2_d)
            else:
                flow_res_f_s, disp_l1_s = self.static_context_networks(torch.cat([x1_out_s, flow_f_s, disp_l1_s], dim=1))
                flow_res_f_d, disp_l1_d = self.dynamic_context_networks(torch.cat([x1_out_d, flow_f_d, disp_l1_d], dim=1))
                flow_res_b_s, disp_l2_s = self.static_context_networks(torch.cat([x2_out_s, flow_b_s, disp_l2_s], dim=1))
                flow_res_b_d, disp_l2_d = self.dynamic_context_networks(torch.cat([x2_out_d, flow_b_d, disp_l2_d], dim=1))

                flow_f_s = flow_f_s + flow_res_f_s
                flow_f_d = flow_f_d + flow_res_f_d
                flow_b_s = flow_b_s + flow_res_b_s
                flow_b_d = flow_b_d + flow_res_b_d

                sceneflows_f_s.append(flow_f_s)
                sceneflows_f_d.append(flow_f_d)
                sceneflows_b_s.append(flow_b_s)
                sceneflows_b_d.append(flow_b_d)
                disps_1_s.append(disp_l1_s)
                disps_1_d.append(disp_l1_d)
                disps_2_s.append(disp_l2_s)                
                disps_2_d.append(disp_l2_d)                
                break

        x1_rev = x1_pyramid[::-1]

        output_dict['flow_f_s'] = upsample_outputs_as(sceneflows_f_s[::-1], x1_rev)
        output_dict['flow_f_d'] = upsample_outputs_as(sceneflows_f_d[::-1], x1_rev)
        output_dict['flow_b_s'] = upsample_outputs_as(sceneflows_b_s[::-1], x1_rev)
        output_dict['flow_b_d'] = upsample_outputs_as(sceneflows_b_d[::-1], x1_rev)

        output_dict['disp_l1_s'] = upsample_outputs_as(disps_1_s[::-1], x1_rev)
        output_dict['disp_l1_d'] = upsample_outputs_as(disps_1_d[::-1], x1_rev)
        output_dict['disp_l2_s'] = upsample_outputs_as(disps_2_s[::-1], x1_rev)
        output_dict['disp_l2_d'] = upsample_outputs_as(disps_2_d[::-1], x1_rev)
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f_s'][ii] = flow_horizontal_flip(output_dict_r['flow_f_s'][ii])
                output_dict_r['flow_f_d'][ii] = flow_horizontal_flip(output_dict_r['flow_f_d'][ii])
                output_dict_r['flow_b_s'][ii] = flow_horizontal_flip(output_dict_r['flow_b_s'][ii])
                output_dict_r['flow_b_d'][ii] = flow_horizontal_flip(output_dict_r['flow_b_d'][ii])
                output_dict_r['disp_l1_s'][ii] = torch.flip(output_dict_r['disp_l1_s'][ii], [3])
                output_dict_r['disp_l1_d'][ii] = torch.flip(output_dict_r['disp_l1_d'][ii], [3])
                output_dict_r['disp_l2_s'][ii] = torch.flip(output_dict_r['disp_l2_s'][ii], [3])
                output_dict_r['disp_l2_d'][ii] = torch.flip(output_dict_r['disp_l2_d'][ii], [3])

            output_dict['output_dict_r'] = output_dict_r

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp_s = []
            flow_f_pp_d = []
            flow_b_pp_s = []
            flow_b_pp_d = []
            disp_l1_pp_s = []
            disp_l1_pp_d = []
            disp_l2_pp_s = []
            disp_l2_pp_d = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp_s.append(post_processing(output_dict['flow_f_s'][ii], flow_horizontal_flip(output_dict_flip['flow_f_s'][ii])))
                flow_f_pp_d.append(post_processing(output_dict['flow_f_d'][ii], flow_horizontal_flip(output_dict_flip['flow_f_d'][ii])))
                flow_b_pp_s.append(post_processing(output_dict['flow_b_s'][ii], flow_horizontal_flip(output_dict_flip['flow_b_s'][ii])))
                flow_b_pp_d.append(post_processing(output_dict['flow_b_d'][ii], flow_horizontal_flip(output_dict_flip['flow_b_d'][ii])))
                disp_l1_pp_s.append(post_processing(output_dict['disp_l1_s'][ii], torch.flip(output_dict_flip['disp_l1_s'][ii], [3])))
                disp_l1_pp_d.append(post_processing(output_dict['disp_l1_d'][ii], torch.flip(output_dict_flip['disp_l1_d'][ii], [3])))
                disp_l2_pp_s.append(post_processing(output_dict['disp_l2_s'][ii], torch.flip(output_dict_flip['disp_l2_s'][ii], [3])))
                disp_l2_pp_d.append(post_processing(output_dict['disp_l2_d'][ii], torch.flip(output_dict_flip['disp_l2_d'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp_s
            output_dict['flow_f_pp'] = flow_f_pp_d
            output_dict['flow_b_pp'] = flow_b_pp_s
            output_dict['flow_b_pp'] = flow_b_pp_d
            output_dict['disp_l1_pp'] = disp_l1_pp_s
            output_dict['disp_l1_pp'] = disp_l1_pp_d
            output_dict['disp_l2_pp'] = disp_l2_pp_s
            output_dict['disp_l2_pp'] = disp_l2_pp_d

        return output_dict
