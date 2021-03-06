from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .correlation_package.correlation import Correlation

from .modules_sceneflow import WarpingLayer_SF
from .modules_sceneflow import initialize_msra, upsample_outputs_as
from .modules_sceneflow import upconv_interpolate as upconv
from .modules_sceneflow import FeatureExtractor, MonoSceneFlowDecoder, ContextNetwork
from .modules_sceneflow import MaskNetDecoder, CameraMotionDecoder, CameraMotionMaskNet
from .modules_sceneflow import apply_rigidity_mask

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import flow_horizontal_flip, post_processing, cm_horizontal_flip


class CNet(nn.Module):
    def __init__(self, args):
        super(CNet, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        self.num_refs = 1

        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_sf = WarpingLayer_SF()

        self.static_flow_estimators = nn.ModuleList()
        self.dynamic_flow_estimators = nn.ModuleList()
        self.mask_decoders = nn.ModuleList()
        self.cam_motion_decoders = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for level, ch in enumerate(self.num_chs[::-1]):
            if level > self.output_level:
                break

            if level == 0:
                num_ch_in = self.dim_corr + ch
                num_ch_in_mask = self.dim_corr + ch + ch + 64
                num_ch_in_cm = self.dim_corr + ch + ch + 32 
            else:
                num_ch_in = self.dim_corr + ch + 64 + 3 + 1
                num_ch_in_mask = self.dim_corr + ch + ch + 64 + 1
                num_ch_in_cm = self.dim_corr + ch + ch + 32 + 6

                self.upconv_layers.append(upconv(64, 64, 3, 2))

            # split decoders
            static_layer_sf = MonoSceneFlowDecoder(num_ch_in)
            dynamic_layer_sf = MonoSceneFlowDecoder(num_ch_in)

            mask_decoder = MaskNetDecoder(num_ch_in_mask)
            cam_motion_decoder = CameraMotionDecoder(num_ch_in_cm)

            self.static_flow_estimators.append(static_layer_sf)
            self.dynamic_flow_estimators.append(dynamic_layer_sf)
            self.mask_decoders.append(mask_decoder)
            self.cam_motion_decoders.append(cam_motion_decoder)

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1,
                            "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}

        self.context_networks = ContextNetwork(64 + 3 + 1 + 6 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, l1_raw, l2_raw, r1_raw, r2_raw, k_l1, k_l2, k_r1, k_r2):

        output_dict = {}

        # on the bottom level are original images
        l1_pyramid = self.feature_pyramid_extractor(l1_raw) + [l1_raw]
        l2_pyramid = self.feature_pyramid_extractor(l2_raw) + [l2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_f_s = []
        sceneflows_f_d = []

        sceneflows_b = []
        sceneflows_b_s = []
        sceneflows_b_d = []

        rigidity_masks_f = []
        rigidity_masks_b = []

        cam_motions_f = []
        cam_motions_b = []

        disps_l1 = []
        disps_l1_s = []
        disps_l1_d = []

        disps_l2 = []
        disps_l2_s = []
        disps_l2_d = []

        for level, (x1, x2) in enumerate(zip(l1_pyramid, l2_pyramid)):

            # warping
            if level == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f = interpolate2d_as(flow_f, x1, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1, mode="bilinear")

                cm_feats_f = interpolate2d_as(cm_feats_f, x1, mode="bilinear")
                cm_feats_b = interpolate2d_as(cm_feats_b, x1, mode="bilinear")

                x1_out = self.upconv_layers[level-1](x1_out)
                x2_out = self.upconv_layers[level-1](x2_out)
                # because K can be changing when doing augmentation
                x2_warp = self.warping_layer_sf(
                    x2, flow_f, disp_l1, k_l1, input_dict['aug_size'])
                x1_warp = self.warping_layer_sf(
                    x1, flow_b, disp_l2, k_l2, input_dict['aug_size'])

            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # joint estimator
            # passing in x1_out flow_f instead of x1_out_s and flow_f_s
            if level == 0:
                x1_out_s, flow_f_s, disp_l1_s = self.static_flow_estimators[level](
                    torch.cat([out_corr_relu_f, x1], dim=1))
                x1_out_d, flow_f_d, disp_l1_d = self.dynamic_flow_estimators[level](
                    torch.cat([out_corr_relu_f, x1], dim=1))
                x1_out = torch.cat([x1_out_s, x1_out_d], dim=1)

                x2_out_s, flow_b_s, disp_l2_s = self.static_flow_estimators[level](
                    torch.cat([out_corr_relu_b, x2], dim=1))
                x2_out_d, flow_b_d, disp_l2_d = self.dynamic_flow_estimators[level](
                    torch.cat([out_corr_relu_b, x2], dim=1))
                x2_out = torch.cat([x2_out_s, x2_out_d], dim=1)

                cm_feats_f, cm_f = self.cam_motion_decoders[level](
                    torch.cat([out_corr_relu_f, x1, x2, x1_out_s], dim=1))
                cm_feats_b, cm_b = self.cam_motion_decoders[level](
                    torch.cat([out_corr_relu_b, x2, x1, x2_out_s], dim=1))

                mask_f_l, mask_f_l_upsampled = self.mask_decoders[level](
                    torch.cat([out_corr_relu_f, x1, x2, x1_out], dim=1))
                mask_b_l, mask_b_l_upsampled = self.mask_decoders[level](
                    torch.cat([out_corr_relu_b, x2, x1, x2_out], dim=1))

                flow_f = apply_rigidity_mask(flow_f_s, flow_f_d, mask_f_l, self._args.mask_thresh)
                flow_b = apply_rigidity_mask(flow_b_s, flow_b_d, mask_b_l, self._args.mask_thresh)
                disp_l1 = apply_rigidity_mask(disp_l1_s, disp_l1_d, mask_f_l, self._args.mask_thresh)
                disp_l2 = apply_rigidity_mask(disp_l2_s, disp_l2_d, mask_b_l, self._args.mask_thresh)

            else:
                x1_out_s, flow_f_s_res, disp_l1_s = self.static_flow_estimators[level](
                    torch.cat([out_corr_relu_f, x1, x1_out, flow_f, disp_l1], dim=1))
                x1_out_d, flow_f_d_res, disp_l1_d = self.dynamic_flow_estimators[level](
                    torch.cat([out_corr_relu_f, x1, x1_out, flow_f, disp_l1], dim=1))
                x1_out = torch.cat([x1_out_s, x1_out_d], dim=1)

                x2_out_s, flow_b_s_res, disp_l2_s = self.static_flow_estimators[level](
                    torch.cat([out_corr_relu_b, x2, x2_out, flow_b, disp_l2], dim=1))
                x2_out_d, flow_b_d_res, disp_l2_d = self.dynamic_flow_estimators[level](
                    torch.cat([out_corr_relu_b, x2, x2_out, flow_b, disp_l2], dim=1))
                x2_out = torch.cat([x2_out_s, x2_out_d], dim=1)

                cm_feats_f, cm_f = self.cam_motion_decoders[level](
                    torch.cat([out_corr_relu_f, x1, x2, x1_out_s, cm_feats_f], dim=1))
                cm_feats_b, cm_b = self.cam_motion_decoders[level](
                    torch.cat([out_corr_relu_b, x2, x1, x2_out_s, cm_feats_b], dim=1))

                mask_f_l, mask_f_l_upsampled = self.mask_decoders[level](
                    torch.cat([out_corr_relu_f, x1, x2, x1_out, mask_f_l_upsampled], dim=1))
                mask_b_l, mask_b_l_upsampled = self.mask_decoders[level](
                    torch.cat([out_corr_relu_b, x2, x1, x2_out, mask_b_l_upsampled], dim=1))

                flow_f_res = apply_rigidity_mask(flow_f_s_res, flow_f_d_res, mask_f_l, self._args.mask_thresh)
                flow_b_res = apply_rigidity_mask(flow_b_s_res, flow_b_d_res, mask_b_l, self._args.mask_thresh)

                disp_l1 = apply_rigidity_mask(disp_l1_s, disp_l1_d, mask_f_l, self._args.mask_thresh)
                disp_l2 = apply_rigidity_mask(disp_l2_s, disp_l2_d, mask_b_l, self._args.mask_thresh)

                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res

            # upsampling or post-processing
            if level != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3

                sceneflows_f.append(flow_f)
                sceneflows_f_s.append(flow_f_s)
                sceneflows_f_d.append(flow_f_d)

                sceneflows_b.append(flow_b)
                sceneflows_b_s.append(flow_b_s)
                sceneflows_b_d.append(flow_b_d)

                disps_l1.append(disp_l1)
                disps_l1_s.append(disp_l1_s)
                disps_l1_d.append(disp_l1_d)

                disps_l2.append(disp_l2)
                disps_l2_s.append(disp_l2_s)
                disps_l2_d.append(disp_l2_d)

                rigidity_masks_f.append(mask_f_l)
                rigidity_masks_b.append(mask_b_l)

                cam_motions_f.append(cm_f)
                cam_motions_b.append(cm_b)

            else:
                # TODO: could feed in decomposed flow here instead
                flow_res_f, disp_l1 = self.context_networks(
                    torch.cat([x1_out, flow_f, disp_l1, cm_feats_f, mask_f_l], dim=1))
                flow_res_b, disp_l2 = self.context_networks(
                    torch.cat([x2_out, flow_b, disp_l2, cm_feats_b, mask_b_l], dim=1))

                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b

                sceneflows_f.append(flow_f)
                sceneflows_f_s.append(flow_f_s)
                sceneflows_f_d.append(flow_f_d)

                sceneflows_b.append(flow_b)
                sceneflows_b_s.append(flow_b_s)
                sceneflows_b_d.append(flow_b_d)

                disps_l1.append(disp_l1)
                disps_l1_s.append(disp_l1_s)
                disps_l1_d.append(disp_l1_d)

                disps_l2.append(disp_l2)
                disps_l2_s.append(disp_l2_s)
                disps_l2_d.append(disp_l2_d)

                rigidity_masks_f.append(mask_f_l)
                rigidity_masks_b.append(mask_b_l)

                cam_motions_f.append(cm_f)
                cam_motions_b.append(cm_b)
                break

        x1_rev = l1_pyramid[::-1]

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

        output_dict['disp_l1'] = upsample_outputs_as(disps_l1[::-1], x1_rev)
        output_dict['disp_l1_s'] = upsample_outputs_as(disps_l1_s[::-1], x1_rev)
        output_dict['disp_l1_d'] = upsample_outputs_as(disps_l1_d[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_l2[::-1], x1_rev)
        output_dict['disp_l2_s'] = upsample_outputs_as(disps_l2_s[::-1], x1_rev)
        output_dict['disp_l2_d'] = upsample_outputs_as(disps_l2_d[::-1], x1_rev)

        output_dict['rigidity_f'] = upsample_outputs_as(rigidity_masks_f[::-1], x1_rev)
        output_dict['rigidity_b'] = upsample_outputs_as(rigidity_masks_b[::-1], x1_rev)

        output_dict['cms_f'] = cam_motions_f[::-1]
        output_dict['cms_b'] = cam_motions_b[::-1]

        return output_dict

    def forward(self, input_dict):

        output_dict = {}

        # Left
        output_dict = self.run_pwc(
            input_dict,
            input_dict['input_l1_aug'],
            input_dict['input_l2_aug'],
            input_dict['input_r1_aug'],
            input_dict['input_r2_aug'],
            input_dict['input_k_l1_aug'],
            input_dict['input_k_l2_aug'],
            input_dict['input_k_r1_aug'],
            input_dict['input_k_r2_aug'])

        # Right
        # ss: train val
        # ft: train
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_r = self.run_pwc(
                input_dict,
                input_l1_flip, input_l2_flip,
                input_r1_flip, input_r2_flip,
                k_l1_flip, k_l2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(
                  output_dict_r['flow_f'][ii])
                output_dict_r['flow_f_s'][ii] = flow_horizontal_flip(
                  output_dict_r['flow_f_s'][ii])
                output_dict_r['flow_f_d'][ii] = flow_horizontal_flip(
                  output_dict_r['flow_f_d'][ii])

                output_dict_r['flow_b'][ii] = flow_horizontal_flip(
                  output_dict_r['flow_b'][ii])
                output_dict_r['flow_b_s'][ii] = flow_horizontal_flip(
                  output_dict_r['flow_b_s'][ii])
                output_dict_r['flow_b_d'][ii] = flow_horizontal_flip(
                  output_dict_r['flow_b_d'][ii])

                output_dict_r['disp_l1'][ii] = torch.flip(
                  output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l1_s'][ii] = torch.flip(
                  output_dict_r['disp_l1_s'][ii], [3])
                output_dict_r['disp_l1_d'][ii] = torch.flip(
                  output_dict_r['disp_l1_d'][ii], [3])

                output_dict_r['disp_l2'][ii] = torch.flip(
                  output_dict_r['disp_l2'][ii], [3])
                output_dict_r['disp_l2_s'][ii] = torch.flip(
                  output_dict_r['disp_l2_s'][ii], [3])
                output_dict_r['disp_l2_d'][ii] = torch.flip(
                  output_dict_r['disp_l2_d'][ii], [3])

                output_dict_r['rigidity_f'][ii] = torch.flip(
                  output_dict_r['rigidity_f'][ii], [3])
                output_dict_r['rigidity_b'][ii] = torch.flip(
                  output_dict_r['rigidity_b'][ii], [3])
                
                output_dict_r['cms_f'][ii] = cm_horizontal_flip(output_dict_r['cms_f'][ii])
                output_dict_r['cms_b'][ii] = cm_horizontal_flip(output_dict_r['cms_b'][ii])

            output_dict['output_dict_r'] = output_dict_r

        # Post Processing
        # ss:           eval
        # ft: train val eval
        if self._args.evaluation or self._args.finetuning:
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


class CNet_CamMaskCombined(nn.Module):
    def __init__(self, args):
        super(CNet_CamMaskCombined, self).__init__()

        self._args = args
        # self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.num_chs = [3, 64, 96, 128, 192, 256, 512]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        self.num_refs = 1

        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_sf = WarpingLayer_SF()

        self.static_flow_estimators = nn.ModuleList()
        self.dynamic_flow_estimators = nn.ModuleList()
        self.cam_mask_decoders = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for level, ch in enumerate(self.num_chs[::-1]):
            if level > self.output_level:
                break

            if level == 0:
                num_ch_in = self.dim_corr + ch + ch
                # num_ch_cm_mask = self.dim_corr + ch + ch + 64 + 32
                num_ch_cm_mask = self.dim_corr + ch + ch
            else:
                num_ch_in = self.dim_corr + ch + ch + 3 + 1
                num_ch_cm_mask = self.dim_corr + ch + ch + 6 + 1

                self.upconv_layers.append(upconv(32, 32, 3, 2))

            # split decoders
            static_layer_sf = MonoSceneFlowDecoderSplit(num_ch_in)
            dynamic_layer_sf = MonoSceneFlowDecoderSplit(num_ch_in)
            cam_mask_decoder = CameraMotionMaskNet(num_ch_cm_mask)

            self.static_flow_estimators.append(static_layer_sf)
            self.dynamic_flow_estimators.append(dynamic_layer_sf)
            self.cam_mask_decoders.append(cam_mask_decoder)

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1,
                            "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}

        self.context_networks = ContextNetwork(32 + 3 + 1 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())
      

    def run_pwc(self, input_dict, l1_raw, l2_raw, r1_raw, r2_raw, k_l1, k_l2, k_r1, k_r2):

        output_dict = {}

        # on the bottom level are original images
        l1_pyramid = self.feature_pyramid_extractor(l1_raw) + [l1_raw]
        l2_pyramid = self.feature_pyramid_extractor(l2_raw) + [l2_raw]

        # outputs
        sceneflows_f_s = []
        sceneflows_f_d = []

        sceneflows_b_s = []
        sceneflows_b_d = []

        rigidity_masks_f = []
        rigidity_masks_b = []

        cam_motions_f = []
        cam_motions_b = []

        disps_l1_s = []
        disps_l1_d = []

        disps_l2_s = []
        disps_l2_d = []

        for level, (x1, x2) in enumerate(zip(l1_pyramid, l2_pyramid)):

            # warping
            if level == 0:
                x2_warp_s = x2
                x2_warp_d = x2
                x1_warp_s = x1
                x1_warp_d = x1
            else:
                flow_f_s = interpolate2d_as(flow_f_s, x1, mode="bilinear")
                flow_f_d = interpolate2d_as(flow_f_d, x1, mode="bilinear")
                flow_b_s = interpolate2d_as(flow_b_s, x1, mode="bilinear")
                flow_b_d = interpolate2d_as(flow_b_d, x1, mode="bilinear")

                disp_l1_s = interpolate2d_as(disp_l1_s, x1, mode="bilinear")
                disp_l1_d = interpolate2d_as(disp_l1_d, x1, mode="bilinear")
                disp_l2_s = interpolate2d_as(disp_l2_s, x1, mode="bilinear")
                disp_l2_d = interpolate2d_as(disp_l2_d, x1, mode="bilinear")

                cm_feats_f = interpolate2d_as(cm_feats_f, x1, mode="bilinear")
                cm_feats_b = interpolate2d_as(cm_feats_b, x1, mode="bilinear")
                mask_f = interpolate2d_as(mask_f, x1, mode="bilinear")
                mask_b = interpolate2d_as(mask_b, x1, mode="bilinear")

                x1_out_s = self.upconv_layers[level-1](x1_out_s)
                x1_out_d = self.upconv_layers[level-1](x1_out_d)
                x2_out_s = self.upconv_layers[level-1](x2_out_s)
                x2_out_d = self.upconv_layers[level-1](x2_out_d)

                # because K can be changing when doing augmentation
                x2_warp_s = self.warping_layer_sf(
                    x2, flow_f_s, disp_l1_s, k_l1, input_dict['aug_size'])
                x2_warp_d = self.warping_layer_sf(
                    x2, flow_f_d, disp_l1_d, k_l1, input_dict['aug_size'])
                x1_warp_s = self.warping_layer_sf(
                    x1, flow_b_s, disp_l2_s, k_l2, input_dict['aug_size'])
                x1_warp_d = self.warping_layer_sf(
                    x1, flow_b_d, disp_l2_d, k_l2, input_dict['aug_size'])

            # correlation
            out_corr_f_s = Correlation.apply(x1, x2_warp_s, self.corr_params)
            out_corr_f_d = Correlation.apply(x1, x2_warp_d, self.corr_params)
            out_corr_b_s = Correlation.apply(x2, x1_warp_s, self.corr_params)
            out_corr_b_d = Correlation.apply(x2, x1_warp_d, self.corr_params)

            out_corr_relu_f_s = self.leakyRELU(out_corr_f_s)
            out_corr_relu_f_d = self.leakyRELU(out_corr_f_d)
            out_corr_relu_b_s = self.leakyRELU(out_corr_b_s)
            out_corr_relu_b_d = self.leakyRELU(out_corr_b_d)

            # joint estimator
            if level == 0:
                x1_out_s, flow_f_s, disp_l1_s = self.static_flow_estimators[level](
                    torch.cat([out_corr_relu_f_s, x1, x2], dim=1))
                x1_out_d, flow_f_d, disp_l1_d = self.dynamic_flow_estimators[level](
                    torch.cat([out_corr_relu_f_d, x1, x2], dim=1))

                x2_out_s, flow_b_s, disp_l2_s = self.static_flow_estimators[level](
                    torch.cat([out_corr_relu_b_s, x2, x1], dim=1))
                x2_out_d, flow_b_d, disp_l2_d = self.dynamic_flow_estimators[level](
                    torch.cat([out_corr_relu_b_d, x2, x1], dim=1))

                cm_feats_f, cm_f, mask_f = self.cam_mask_decoders[level](
                  torch.cat([out_corr_relu_f_s, x1, x2], dim=1))
                cm_feats_b, cm_b, mask_b = self.cam_mask_decoders[level](
                  torch.cat([out_corr_relu_b_d, x2, x1], dim=1))

            else:
                x1_out_f_s, flow_f_s_res, disp_l1_s = self.static_flow_estimators[level](
                    torch.cat([out_corr_relu_f_s, x1, x2, flow_f_s, disp_l1_s], dim=1))
                x1_out_f_d, flow_f_d_res, disp_l1_d = self.dynamic_flow_estimators[level](
                    torch.cat([out_corr_relu_f_d, x1, x2, flow_f_d, disp_l1_d], dim=1))

                x2_out_b_s, flow_b_s_res, disp_l2_s = self.static_flow_estimators[level](
                    torch.cat([out_corr_relu_b_s, x2, x1, flow_b_s, disp_l2_s], dim=1))
                x2_out_b_d, flow_b_d_res, disp_l2_d = self.dynamic_flow_estimators[level](
                    torch.cat([out_corr_relu_b_d, x2, x1, flow_b_d, disp_l2_d], dim=1))

                cm_feats_f, cm_f, mask_f = self.cam_mask_decoders[level](
                  torch.cat([out_corr_relu_f_s, x1, x2, cm_feats_f, mask_f], dim=1))
                cm_feats_b, cm_b, mask_b = self.cam_mask_decoders[level](
                  torch.cat([out_corr_relu_b_s, x2, x1, cm_feats_b, mask_b], dim=1))

                flow_f_s = flow_f_s + flow_f_s_res
                flow_f_d = flow_f_d + flow_f_d_res
                flow_b_s = flow_b_s + flow_b_s_res
                flow_b_d = flow_b_d + flow_b_d_res

            # upsampling or post-processing
            if level != self.output_level:
                disp_l1_s = self.sigmoid(disp_l1_s) * 0.3
                disp_l1_d = self.sigmoid(disp_l1_d) * 0.3
                disp_l2_s = self.sigmoid(disp_l2_s) * 0.3
                disp_l2_d = self.sigmoid(disp_l2_d) * 0.3

                sceneflows_f_s.append(flow_f_s)
                sceneflows_f_d.append(flow_f_d)
                sceneflows_b_s.append(flow_b_s)
                sceneflows_b_d.append(flow_b_d)
                disps_l1_s.append(disp_l1_s)
                disps_l1_d.append(disp_l1_d)
                disps_l2_s.append(disp_l2_s)
                disps_l2_d.append(disp_l2_d)
                rigidity_masks_f.append(mask_f)
                rigidity_masks_b.append(mask_b)
                cam_motions_f.append(cm_f)
                cam_motions_b.append(cm_b)

            else:
                flow_res_f_s, disp_l1_s = self.context_networks(
                    torch.cat([x1_out_s, flow_f_s, disp_l1_s, mask_f], dim=1))
                flow_res_f_d, disp_l1_d = self.context_networks(
                    torch.cat([x1_out_d, flow_f_d, disp_l1_d, mask_f], dim=1))
                flow_res_b_s, disp_l2_s = self.context_networks(
                    torch.cat([x2_out_s, flow_b_s, disp_l2_s, mask_b], dim=1))
                flow_res_b_d, disp_l2_d = self.context_networks(
                    torch.cat([x2_out_d, flow_b_d, disp_l2_d, mask_b], dim=1))

                flow_f_s = flow_f_s + flow_res_f_s
                flow_f_d = flow_f_d + flow_res_f_d
                flow_b_s = flow_b_s + flow_res_b_s
                flow_b_d = flow_b_d + flow_res_b_d

                sceneflows_f_s.append(flow_f_s)
                sceneflows_f_d.append(flow_f_d)
                sceneflows_b_s.append(flow_b_s)
                sceneflows_b_d.append(flow_b_d)
                disps_l1_s.append(disp_l1_s)
                disps_l1_d.append(disp_l1_d)
                disps_l2_s.append(disp_l2_s)
                disps_l2_d.append(disp_l2_d)
                rigidity_masks_f.append(mask_f)
                rigidity_masks_b.append(mask_b)
                cam_motions_f.append(cm_f)
                cam_motions_b.append(cm_b)
                break

        x1_rev = l1_pyramid[::-1]

        output_dict['flow_f_s'] = upsample_outputs_as(
            sceneflows_f_s[::-1], x1_rev)
        output_dict['flow_f_d'] = upsample_outputs_as(
            sceneflows_f_d[::-1], x1_rev)

        output_dict['flow_b_s'] = upsample_outputs_as(
            sceneflows_b_s[::-1], x1_rev)
        output_dict['flow_b_d'] = upsample_outputs_as(
            sceneflows_b_d[::-1], x1_rev)

        output_dict['disp_l1_s'] = upsample_outputs_as(disps_l1_s[::-1], x1_rev)
        output_dict['disp_l1_d'] = upsample_outputs_as(disps_l1_d[::-1], x1_rev)
        output_dict['disp_l2_s'] = upsample_outputs_as(disps_l2_s[::-1], x1_rev)
        output_dict['disp_l2_d'] = upsample_outputs_as(disps_l2_d[::-1], x1_rev)

        output_dict['rigidity_f'] = upsample_outputs_as(rigidity_masks_f[::-1], x1_rev)
        output_dict['rigidity_b'] = upsample_outputs_as(rigidity_masks_b[::-1], x1_rev)

        output_dict['cms_f'] = cam_motions_f[::-1]
        output_dict['cms_b'] = cam_motions_b[::-1]

        return output_dict

    def forward(self, input_dict):

        output_dict = {}

        # Left
        output_dict = self.run_pwc(
            input_dict,
            input_dict['input_l1_aug'],
            input_dict['input_l2_aug'],
            input_dict['input_r1_aug'],
            input_dict['input_r2_aug'],
            input_dict['input_k_l1_aug'],
            input_dict['input_k_l2_aug'],
            input_dict['input_k_r1_aug'],
            input_dict['input_k_r2_aug'])

        # Right
        # ss: train val
        # ft: train
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_r = self.run_pwc(
                input_dict,
                input_l1_flip, input_l2_flip,
                input_r1_flip, input_r2_flip,
                k_l1_flip, k_l2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f_s'])):
                output_dict_r['flow_f_s'][ii] = flow_horizontal_flip(
                  output_dict_r['flow_f_s'][ii])
                output_dict_r['flow_f_d'][ii] = flow_horizontal_flip(
                  output_dict_r['flow_f_d'][ii])

                output_dict_r['flow_b_s'][ii] = flow_horizontal_flip(
                  output_dict_r['flow_b_s'][ii])
                output_dict_r['flow_b_d'][ii] = flow_horizontal_flip(
                  output_dict_r['flow_b_d'][ii])

                output_dict_r['disp_l1_s'][ii] = torch.flip(
                  output_dict_r['disp_l1_s'][ii], [3])
                output_dict_r['disp_l1_d'][ii] = torch.flip(
                  output_dict_r['disp_l1_d'][ii], [3])

                output_dict_r['disp_l2_s'][ii] = torch.flip(
                  output_dict_r['disp_l2_s'][ii], [3])
                output_dict_r['disp_l2_d'][ii] = torch.flip(
                  output_dict_r['disp_l2_d'][ii], [3])

                output_dict_r['rigidity_f'][ii] = torch.flip(
                  output_dict_r['rigidity_f'][ii], [3])
                output_dict_r['rigidity_b'][ii] = torch.flip(
                  output_dict_r['rigidity_b'][ii], [3])
                
                output_dict_r['cms_f'][ii] = cm_horizontal_flip(output_dict_r['cms_f'][ii])
                output_dict_r['cms_b'][ii] = cm_horizontal_flip(output_dict_r['cms_b'][ii])

            output_dict['output_dict_r'] = output_dict_r

        # Post Processing
        # ss:           eval
        # ft: train val eval
        if self._args.evaluation or self._args.finetuning:
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
