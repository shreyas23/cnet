from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging

from .correlation_package.correlation import Correlation

from .modules_sceneflow import get_grid, WarpingLayer_SF
from .modules_sceneflow import initialize_msra, upsample_outputs_as
from .modules_sceneflow import upconv_interpolate as upconv
from .modules_sceneflow import FeatureExtractor, MonoSceneFlowDecoder, MaskNetDecoder, ContextNetwork
from .modules_sceneflow import apply_rigidity_mask

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import flow_horizontal_flip, intrinsic_scale, get_pixelgrid, post_processing


class CNet(nn.Module):
    def __init__(self, args):
        super(CNet, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7

        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_sf = WarpingLayer_SF()

        self.static_flow_estimators = nn.ModuleList()
        self.dynamic_flow_estimators = nn.ModuleList()
        self.mask_decoders = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch
                # out_corr_relu, x, x_out (64 s+d), flow_s, flow_d, disp_s, disp_d
                num_ch_in_mask = self.dim_corr + ch + 64 + 3 + 3 + 1 + 1
            else:
                num_ch_in = self.dim_corr + ch + 64 + 3 + 1
                # out_corr_relu, x, x_out(s & d combined -> 64 ch instead of 32), flow_res, flow_res, disp_s, disp_d + rigidity_mask_upsampled
                num_ch_in_mask = self.dim_corr + ch + 64 + 3 + 3 + 1 + 1 + 1
                self.upconv_layers.append(upconv(64, 64, 3, 2))

            # split decoders
            static_layer_sf = MonoSceneFlowDecoder(num_ch_in)
            dynamic_layer_sf = MonoSceneFlowDecoder(num_ch_in)
            mask_decoder = MaskNetDecoder(num_ch_in_mask)

            self.static_flow_estimators.append(static_layer_sf)
            self.dynamic_flow_estimators.append(dynamic_layer_sf)
            self.mask_decoders.append(mask_decoder)

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1,
                            "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}

        self.context_networks = ContextNetwork(64 + 3 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):

        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_f_s = []
        sceneflows_f_d = []

        sceneflows_b = []
        sceneflows_b_s = []
        sceneflows_b_d = []

        disps_1 = []
        disps_1_s = []
        disps_1_d = []

        disps_2 = []
        disps_2_s = []
        disps_2_d = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f = interpolate2d_as(flow_f, x1, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1, mode="bilinear")
                x1_out = self.upconv_layers[l-1](x1_out)
                x2_out = self.upconv_layers[l-1](x2_out)
                # becuase K can be changing when doing augmentation
                x2_warp = self.warping_layer_sf(
                    x2, flow_f, disp_l1, k1, input_dict['aug_size'])
                x1_warp = self.warping_layer_sf(
                    x1, flow_b, disp_l2, k2, input_dict['aug_size'])

            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # joint estimator
            # passing in x1_out flow_f instead of x1_out_s and flow_f_s
            if l == 0:
                x1_out_s, flow_f_s, disp_l1_s = self.static_flow_estimators[l](
                    torch.cat([out_corr_relu_f, x1], dim=1))
                x1_out_d, flow_f_d, disp_l1_d = self.dynamic_flow_estimators[l](
                    torch.cat([out_corr_relu_f, x1], dim=1))
                x1_out = torch.cat([x1_out_s, x1_out_d], dim=1)

                x2_out_s, flow_b_s, disp_l2_s = self.static_flow_estimators[l](
                    torch.cat([out_corr_relu_b, x2], dim=1))
                x2_out_d, flow_b_d, disp_l2_d = self.dynamic_flow_estimators[l](
                    torch.cat([out_corr_relu_b, x2], dim=1))
                x2_out = torch.cat([x2_out_s, x2_out_d], dim=1)

                import numpy as np
                rigidity_mask_fwd, rigidity_mask_fwd_upsampled = self.mask_decoders[l](
                    torch.cat([out_corr_relu_f, x1, x1_out, flow_f_s, flow_f_d, disp_l1_s, disp_l1_d], dim=1))

                rigidity_mask_bwd, rigidity_mask_bwd_upsampled = self.mask_decoders[l](
                    torch.cat([out_corr_relu_b, x2, x2_out, flow_b_s, flow_b_d, disp_l2_s, disp_l2_d], dim=1))

                flow_f = apply_rigidity_mask(
                    flow_f_s, flow_f_d, rigidity_mask_fwd)
                flow_b = apply_rigidity_mask(
                    flow_b_s, flow_b_d, rigidity_mask_bwd)
                disp_l1 = apply_rigidity_mask(
                    disp_l1_s, disp_l1_d, rigidity_mask_fwd)
                disp_l2 = apply_rigidity_mask(
                    disp_l2_s, disp_l2_d, rigidity_mask_bwd)

            else:
                x1_out_s, flow_f_s_res, disp_l1_s = self.static_flow_estimators[l](
                    torch.cat([out_corr_relu_f, x1, x1_out, flow_f, disp_l1], dim=1))
                x1_out_d, flow_f_d_res, disp_l1_d = self.dynamic_flow_estimators[l](
                    torch.cat([out_corr_relu_f, x1, x1_out, flow_f, disp_l1], dim=1))
                x1_out = torch.cat([x1_out_s, x1_out_d], dim=1)

                x2_out_s, flow_b_s_res, disp_l2_s = self.static_flow_estimators[l](
                    torch.cat([out_corr_relu_b, x2, x2_out, flow_b, disp_l2], dim=1))
                x2_out_d, flow_b_d_res, disp_l2_d = self.dynamic_flow_estimators[l](
                    torch.cat([out_corr_relu_b, x2, x2_out, flow_b, disp_l2], dim=1))
                x2_out = torch.cat([x2_out_s, x2_out_d], dim=1)

                if self._args['debug']:
                    print([x[1] for x in [out_corr_relu_f.shape, x1.shape, x1_out.shape, flow_f_s_res.shape,
                                          flow_f_d_res.shape, disp_l1_s.shape, disp_l1_d.shape, rigidity_mask_fwd_upsampled.shape]])
                rigidity_mask_fwd, rigidity_mask_fwd_upsampled = self.mask_decoders[l](
                    torch.cat([out_corr_relu_f, x1, x1_out, flow_f_s_res, flow_f_d_res, disp_l1_s, disp_l1_d, rigidity_mask_fwd_upsampled], dim=1))

                rigidity_mask_bwd, rigidity_mask_bwd_upsampled = self.mask_decoders[l](
                    torch.cat([out_corr_relu_b, x2, x2_out, flow_b_s_res, flow_b_d_res, disp_l2_s, disp_l2_d, rigidity_mask_bwd_upsampled], dim=1))

                flow_f_res = apply_rigidity_mask(
                    flow_f_s_res, flow_f_d_res, rigidity_mask_fwd)
                flow_b_res = apply_rigidity_mask(
                    flow_b_s_res, flow_b_d_res, rigidity_mask_bwd)
                disp_l1 = apply_rigidity_mask(
                    disp_l1_s, disp_l1_d, rigidity_mask_fwd)
                disp_l2 = apply_rigidity_mask(
                    disp_l2_s, disp_l2_d, rigidity_mask_bwd)

                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3

                sceneflows_f.append(flow_f)
                sceneflows_f_s.append(flow_f_s)
                sceneflows_f_d.append(flow_f_d)

                sceneflows_b.append(flow_b)
                sceneflows_b_s.append(flow_b_s)
                sceneflows_b_d.append(flow_b_d)

                disps_1.append(disp_l1)
                disps_1_s.append(disp_l1_s)
                disps_1_d.append(disp_l1_d)
                disps_2.append(disp_l2)
                disps_2_s.append(disp_l2_s)
                disps_2_d.append(disp_l2_d)

            else:
                # TODO: could feed in decomposed flow here instead
                flow_res_f, disp_l1 = self.context_networks(
                    torch.cat([x1_out, flow_f, disp_l1], dim=1))
                flow_res_b, disp_l2 = self.context_networks(
                    torch.cat([x2_out, flow_b, disp_l2], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                break

        x1_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_f_s'] = upsample_outputs_as(
            sceneflows_f_s[::-1], x1_rev)
        output_dict['flow_f_d'] = upsample_outputs_as(
            sceneflows_f_d[::-1], x1_rev)

        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['flow_b_s'] = upsample_outputs_as(
            sceneflows_b_s[::-1], x1_rev)
        output_dict['flow_b_d'] = upsample_outputs_as(
            sceneflows_b_d[::-1], x1_rev)

        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)

        return output_dict

    def forward(self, input_dict):

        output_dict = {}

        # Left
        output_dict = self.run_pwc(
            input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

        # Right
        # ss: train val
        # ft: train
        if self.training or (not self._args['finetuning'] and not self._args['evaluation']):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(
                input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(
                    output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(
                    output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(
                    output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(
                    output_dict_r['disp_l2'][ii], [3])

            output_dict['output_dict_r'] = output_dict_r

        # Post Processing
        # ss:           eval
        # ft: train val eval
        if self._args['evaluation'] or self._args['finetuning']:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(
                input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(
                    output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(
                    output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(
                    output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(
                    output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict
