from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf

from models.forwardwarp_package.forward_warp import forward_warp
from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms, reconstructImg, reconstructPts, projectSceneFlow2Flow
from utils.monodepth_eval import compute_errors
from models.modules_sceneflow import WarpingLayer_Flow

from utils.sceneflow_util import disp2depth_kitti, flow_horizontal_flip
from utils.inverse_warp import flow_warp, pose2flow, pose_vec2mat, inverse_warp

import sys

###############################################
# Basic Module
###############################################

def _reconstruction_error(tgt_img, ref_img_warped, ssim_w):
    diff = (_elementwise_l1(tgt_img, ref_img_warped) * (1.0 - ssim_w) +
            _SSIM(tgt_img, ref_img_warped) * ssim_w).mean(dim=1, keepdim=True)
    return diff


def _elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=2, dim=1, keepdim=True)


def _elementwise_l1(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=1, dim=1, keepdim=True)


def _elementwise_robust_epe_char(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.pow(torch.norm(residual, p=2, dim=1, keepdim=True) + 0.01, 0.4)


def _SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    SSIM_img = torch.clamp((1 - SSIM) / 2, 0, 1)

    return tf.pad(SSIM_img, pad=(1, 1, 1, 1), mode='constant', value=0)


def _apply_disparity(img, disp):
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(
        batch_size, height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(
        batch_size, width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    # Disparity is passed in NCHW format with 1 channel
    x_shifts = disp[:, 0, :, :]
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = tf.grid_sample(img, 2 * flow_field - 1,
                            mode='bilinear', padding_mode='zeros')

    return output


def _generate_image_left(img, disp):
    return _apply_disparity(img, -disp)


def _adaptive_disocc_detection(flow):

    # init mask
    b, _, h, w, = flow.size()
    mask = torch.ones(b, 1, h, w, dtype=flow.dtype,
                      device=flow.device).float().requires_grad_(False)
    flow = flow.transpose(1, 2).transpose(2, 3)

    disocc = torch.clamp(forward_warp()(mask, flow), 0, 1)
    disocc_map = (disocc > 0.5)

    if disocc_map.float().sum() < (b * h * w / 2):
        disocc_map = torch.ones(
            b, 1, h, w, dtype=torch.bool, device=flow.device).requires_grad_(False)

    return disocc_map


def _adaptive_disocc_detection_disp(disp):

    # # init
    b, _, h, w, = disp.size()
    mask = torch.ones(b, 1, h, w, dtype=disp.dtype,
                      device=disp.device).float().requires_grad_(False)
    flow = torch.zeros(b, 2, h, w, dtype=disp.dtype,
                       device=disp.device).float().requires_grad_(False)
    flow[:, 0:1, :, :] = disp * w
    flow = flow.transpose(1, 2).transpose(2, 3)

    disocc = torch.clamp(forward_warp()(mask, flow), 0, 1)
    disocc_map = (disocc > 0.5)

    if disocc_map.float().sum() < (b * h * w / 2):
        disocc_map = torch.ones(
            b, 1, h, w, dtype=torch.bool, device=disp.device).requires_grad_(False)

    return disocc_map


def _gradient_x(img):
    img = tf.pad(img, (0, 1, 0, 0), mode="replicate")
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
    return gx


def _gradient_y(img):
    img = tf.pad(img, (0, 0, 0, 1), mode="replicate")
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
    return gy


def _gradient_x_2nd(img):
    img_l = tf.pad(img, (1, 0, 0, 0), mode="replicate")[:, :, :, :-1]
    img_r = tf.pad(img, (0, 1, 0, 0), mode="replicate")[:, :, :, 1:]
    gx = img_l + img_r - 2 * img
    return gx


def _gradient_y_2nd(img):
    img_t = tf.pad(img, (0, 0, 1, 0), mode="replicate")[:, :, :-1, :]
    img_b = tf.pad(img, (0, 0, 0, 1), mode="replicate")[:, :, 1:, :]
    gy = img_t + img_b - 2 * img
    return gy


def _smoothness_motion_2nd(sf, img, beta=1):
    sf_grad_x = _gradient_x_2nd(sf)
    sf_grad_y = _gradient_y_2nd(sf)

    img_grad_x = _gradient_x(img)
    img_grad_y = _gradient_y(img)
    weights_x = torch.exp(-torch.mean(torch.abs(img_grad_x),
                                      1, keepdim=True) * beta)
    weights_y = torch.exp(-torch.mean(torch.abs(img_grad_y),
                                      1, keepdim=True) * beta)

    smoothness_x = sf_grad_x * weights_x
    smoothness_y = sf_grad_y * weights_y

    return (smoothness_x.abs() + smoothness_y.abs())


def _disp2depth_kitti_K(disp, k_value):

    mask = (disp > 0).float()
    depth = k_value.unsqueeze(1).unsqueeze(
        1).unsqueeze(1) * 0.54 / (disp + (1.0 - mask))

    return depth


def _depth2disp_kitti_K(depth, k_value):

    disp = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / depth

    return disp

###############################################
## Self Supervised Consistency Loss Function ## 
###############################################

class Loss_SceneFlow_SelfSup_Consistency(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_Consistency, self).__init__()

        self._args = args
        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_smooth_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200

        self._num_views = args.num_views
        self._consistency_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self._flow_diff_thresh = 0.01

    def consistency_loss(self, 
                         sfs_f, sfs_b,
                         cms_f, cms_b,
                         disps_1, disps_2,
                         disp_occs_1, disp_occs_2,
                         imgs_1_aug, imgs_2_aug,
                         rigidity_mask_f, rigidity_mask_b,
                         k_1_aug, k_2_aug,
                         cam_r2l, aug_size, baseline):

        # pts_1_tfs =   []
        # pts_2_tfs =   []
        # pts_1_warp =  []
        # pts_2_warp =  []

        ## format: [left, right] ## 
        imgs_1_warp = [] 
        imgs_2_warp = []
        occ_maps_f =  []
        occ_maps_b =  []

        _, _, h_dp, w_dp = sfs_f[0].size()
        disps_1 = [disps_1[i] * w_dp for i in range(self._num_views)]
        disps_2 = [disps_2[i] * w_dp for i in range(self._num_views)]

        # scale
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp

        rigidity_mask_bool_f = rigidity_mask_f >= self._args.mask_thresh

        for i in range(self._num_views):
          pts1, k1_scale = pixel2pts_ms(k_1_aug[i], disps_1[i], local_scale / aug_size)
          pts2, k2_scale = pixel2pts_ms(k_2_aug[i], disps_2[i], local_scale / aug_size)

          _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sfs_f[i], [h_dp, w_dp])
          _, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sfs_b[i], [h_dp, w_dp])

          pts2_warp = reconstructPts(coord1, pts2)
          pts1_warp = reconstructPts(coord2, pts1)

          flow_f = projectSceneFlow2Flow(k1_scale, sfs_f[i], disps_1[i])
          flow_b = projectSceneFlow2Flow(k2_scale, sfs_b[i], disps_2[i])
          occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occs_2[i]
          occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occs_1[i]

          # Image reconstruction loss
          img_2_warp = reconstructImg(coord1, imgs_2_aug[i])
          img_1_warp = reconstructImg(coord2, imgs_1_aug[i])

          # pts_1_tfs.append(pts1_tf)
          # pts_2_tfs.append(pts2_tf)
          # pts_1_warps.append(pts1_warp)
          # pts_2_warps.append(pts2_warp)
          occ_maps_f.append(occ_map_f)
          occ_maps_b.append(occ_map_b)
          imgs_1_warp.append(img_1_warp)
          imgs_2_warp.append(img_2_warp)

        # cross term image reconstruction loss
        cross_warp_img_f = _apply_disparity(imgs_1_warp[0], disps_2[0])
        cross_warp_diff_f = _reconstruction_error(imgs_2_aug[1], cross_warp_img_f, self._ssim_w)

        cross_warp_img_b = _apply_disparity(imgs_2_warp[0], disps_1[0])
        cross_warp_diff_b = _reconstruction_error(imgs_1_aug[1], cross_warp_img_b, self._ssim_w)

        loss_cross_f = cross_warp_diff_f[occ_maps_f[0]].mean()
        loss_cross_b = cross_warp_diff_b[occ_maps_b[0]].mean()

        cross_warp_diff_f[~occ_maps_f[0]].detach_()
        cross_warp_diff_b[~occ_maps_b[0]].detach_()

        stereo_consistency_loss = loss_cross_f + loss_cross_b
        assert (not torch.isnan(stereo_consistency_loss)), "stereo consistency loss is nan"

        # egomotion consistency loss
        depth_l1 = disp2depth_kitti(disps_1[0], k_1_aug[0][:, 0, 0], baseline=baseline).squeeze(1)
        depth_l2 = disp2depth_kitti(disps_2[0], k_2_aug[0][:, 0, 0], baseline=baseline).squeeze(1)
        depth_r1 = disp2depth_kitti(disps_1[1], k_1_aug[1][:, 0, 0], baseline=baseline).squeeze(1)

        cam_flow_l = pose2flow(depth_l1, cms_f[0], k_1_aug[0], torch.inverse(k_1_aug[0]))
        cam_flow_l_b = pose2flow(depth_l2, cms_b[0], k_2_aug[0], torch.inverse(k_2_aug[0]))

        cam_flow_r = pose2flow(depth_r1, cms_f[1], k_1_aug[1], torch.inverse(k_1_aug[1]))
        cam_flow_r_flip = pose2flow(depth_r1, cms_f[1], k_1_aug[1], torch.inverse(k_1_aug[1]), cam_tform=cam_r2l)
        cam_flow_r_warp = flow_horizontal_flip(cam_flow_r_flip)

        ego_consistency_loss = _elementwise_epe(cam_flow_l, cam_flow_r_warp).mean() 
        assert (not torch.isnan(ego_consistency_loss)), "ego consistency loss is nan"

        ##################################################################
        # static reconstruction loss through camera motion and disparity #
        # FOR LEFT IMAGES ONLY                                           #
        ##################################################################

        # static_warp_l1_f = inverse_warp(imgs_1_aug[0], depth_l1, cms_f[0], k_1_aug[0], torch.inverse(k_1_aug[0]))
        # static_warp_r1_f = inverse_warp(img_r1_aug, depth_r1, cms_f[0], k_r1_aug, torch.inverse(k_r1_aug))
        static_warp_l1_f = flow_warp(imgs_1_aug[0], cam_flow_l)
        static_warp_l1_error = _reconstruction_error(imgs_2_aug[0], static_warp_l1_f, self._ssim_w)
        # static_warp_r1_f = flow_warp(img_1_aug[1], cam_flow_r)
        # static_warp_r1_error = _reconstruction_error(img_2_aug[1], static_warp_r1_f, self._ssim_w)
        static_warp_error = static_warp_l1_error # + static_warp_r1_error
        cam_occ_map_l_f = _adaptive_disocc_detection(cam_flow_l_b)
        # cam_occ_map_r_f = _adaptive_disocc_detection(cam_flow_r)

        if (~rigidity_mask_bool_f).sum() != 0:
          cam_motion_loss = static_warp_error[~rigidity_mask_bool_f & cam_occ_map_l_f].mean() 
        if torch.isnan(cam_motion_loss):
          cam_motion_loss = static_warp_error[cam_occ_map_l_f].mean() 

        assert (not torch.isnan(cam_motion_loss)), f"cam motion loss is nan: {static_warp_error.mean()} {(cam_occ_map_l_f).sum()}"
        
        # mask consensus loss
        flow_f = projectSceneFlow2Flow(k_1_aug[0], sfs_f[0], disps_1[0])
        img_l2_reconstruction_error = _reconstruction_error(imgs_2_aug[0], imgs_1_warp[0], self._ssim_w)
        target_mask = (static_warp_l1_error <= img_l2_reconstruction_error).bool()
        flow_diff = (_elementwise_epe(cam_flow_l, flow_f) < self._flow_diff_thresh).bool()
        census_target_mask = target_mask | flow_diff

        mask_consensus_loss = tf.binary_cross_entropy(rigidity_mask_f, census_target_mask.float())
        assert (not torch.isnan(mask_consensus_loss)), "mask consensus loss is nan"

        # induced static scene flow knowledge distillation loss
        static_flow_consistency_diff = _elementwise_epe(cam_flow_l, flow_f)
        static_flow_consistency_loss = static_flow_consistency_diff[~rigidity_mask_bool_f].mean()
        assert (not torch.isnan(static_flow_consistency_loss)), f"static flow consistency loss is nan: {static_flow_consistency_diff.mean()}, {~rigidity_mask_bool_f.sum()}"

        total_loss = [stereo_consistency_loss, 
                    #  ego_consistency_loss,
                      0,
                      static_flow_consistency_loss,
                      cam_motion_loss,
                      mask_consensus_loss]

        total_loss = sum([w * l for w, l in zip(self._consistency_weights, total_loss)])

        return total_loss, \
               stereo_consistency_loss, \
               ego_consistency_loss, \
               static_flow_consistency_loss, \
               cam_motion_loss, \
               mask_consensus_loss

    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        # Photometric loss
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) +
                    _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_img = (img_diff[left_occ]).mean()
        img_diff[~left_occ].detach_()

        # Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(
            disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_smooth_w * loss_smooth, left_occ

    def sceneflow_loss(self,
                       sf_f, sf_b,
                       disp_l1, disp_l2,
                       disp_occ_l1, disp_occ_l2,
                       img_l1_aug, img_l2_aug,
                       k_l1_aug, k_l2_aug,
                       aug_size, ii):

        _, _, h_dp, w_dp = sf_f.size()
        disp_l1 = disp_l1 * w_dp
        disp_l2 = disp_l2 * w_dp

        # scale
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp

        pts1, k1_scale = pixel2pts_ms(
            k_l1_aug, disp_l1, local_scale / aug_size)
        pts2, k2_scale = pixel2pts_ms(
            k_l2_aug, disp_l2, local_scale / aug_size)

        _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
        _, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp])

        pts2_warp = reconstructPts(coord1, pts2)
        pts1_warp = reconstructPts(coord2, pts1)

        flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
        flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
        occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

        # Image reconstruction loss
        img_l2_warp = reconstructImg(coord1, img_l2_aug)
        img_l1_warp = reconstructImg(coord2, img_l1_aug)

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) +
                     _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) +
                     _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)

        loss_im1 = img_diff1[occ_map_f].mean()
        loss_im2 = img_diff2[occ_map_b].mean()

        img_diff1[~occ_map_f].detach_()
        img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2

        if self._args.debugging:
          print(f"num pixels: {occ_map_b.sum()}, {occ_map_f.sum()}")
          print(f"iter {ii}: sf loss = {loss_im}")

        # Point reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(
            dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(
            dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        loss_pts1 = pts_diff1[occ_map_f].mean()
        loss_pts2 = pts_diff2[occ_map_b].mean()

        pts_diff1[~occ_map_f].detach_()
        pts_diff2[~occ_map_b].detach_()

        loss_pts = loss_pts1 + loss_pts2

        # 3D motion smoothness loss
        loss_3d_s = ((_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() +
                     (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean()) / (2 ** ii)

        # Loss Summation
        sceneflow_loss = loss_im + self._sf_3d_pts * \
            loss_pts + self._sf_3d_sm * loss_3d_s

        return sceneflow_loss, loss_im, loss_pts, loss_3d_s

    def detaching_grad_of_outputs(self, output_dict):

        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_f_s'][ii].detach_()
            output_dict['flow_f_d'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['flow_b_s'][ii].detach_()
            output_dict['flow_b_d'][ii].detach_()
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l1_s'][ii].detach_()
            output_dict['disp_l1_d'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()
            output_dict['disp_l2_s'][ii].detach_()
            output_dict['disp_l2_d'][ii].detach_()
            output_dict['cms_f'][ii].detach_()
            output_dict['cms_b'][ii].detach_()
            output_dict['rigidity_f'][ii].detach_()
            output_dict['rigidity_b'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        # batch_size = target_dict['input_l1'].size(0)
        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        loss_sf_3d = 0
        loss_sf_sm = 0

        loss_cons_sum = 0
        loss_cm_sum = 0
        loss_cross_sum = 0
        loss_ego_sum = 0
        loss_mask_sum = 0
        loss_static_sum = 0

        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        k_r1_aug = target_dict['input_k_r1_aug']
        k_r2_aug = target_dict['input_k_r2_aug']
        aug_size = target_dict['aug_size']
        cam_r2l = target_dict['input_cam_r2l']
        baseline = target_dict['baseline'].item()

        sfs_f_r = output_dict['output_dict_r']['flow_f']
        sfs_b_r = output_dict['output_dict_r']['flow_b']
        cms_f_r = output_dict['output_dict_r']['cms_f']
        cms_b_r = output_dict['output_dict_r']['cms_b']
        disp_r1 = output_dict['output_dict_r']['disp_l1']
        disp_r2 = output_dict['output_dict_r']['disp_l2']

        for ii, (sf_f_l, sf_b_l, sf_f_r, sf_b_r,
                 cms_f_l, cms_b_l, cms_f_r, cms_b_r,
                 rigidity_mask_f, rigidity_mask_b,
                 disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], sfs_f_r, sfs_b_r,
                                                    output_dict['cms_f'], output_dict['cms_b'], cms_f_r, cms_b_r,
                                                    output_dict['rigidity_f'], output_dict['rigidity_b'],
                                                    output_dict['disp_l1'], output_dict['disp_l2'], disp_r1, disp_r2)):

            assert(sf_f_l.size()[2:4] == sf_b_l.size()[2:4])
            assert(sf_f_l.size()[2:4] == sf_f_r.size()[2:4])
            assert(sf_f_l.size()[2:4] == sf_b_l.size()[2:4])
            assert(sf_f_l.size()[2:4] == rigidity_mask_f.size()[2:4])
            assert(sf_f_l.size()[2:4] == disp_l1.size()[2:4])
            assert(sf_f_l.size()[2:4] == disp_l2.size()[2:4])

            # For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f_l)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b_l)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f_l)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b_l)

            # Disp Loss
            loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(
                disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(
                disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + \
                (loss_disp_l1 + loss_disp_l2) * self._weights[ii]

            # Sceneflow Loss
            loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f_l, sf_b_l,  # sf: [static, dynamic]
                                                                               disp_l1, disp_l2,
                                                                               disp_occ_l1, disp_occ_l2,
                                                                               img_l1_aug, img_l2_aug,
                                                                               k_l1_aug, k_l2_aug,
                                                                               aug_size, ii)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]
            loss_sf_2d = loss_sf_2d + loss_im
            loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s

            imgs_1_aug = [img_l1_aug, img_r1_aug]
            imgs_2_aug = [img_l2_aug, img_r2_aug]

            sfs_f = [sf_f_l, sf_b_r]
            sfs_b = [sf_b_l, sf_b_r]
            cms_f = [cms_f_l, cms_f_r]
            cms_b = [cms_b_l, cms_b_r]
            disps_1 = [disp_l1, disp_r1]
            disps_2 = [disp_l2, disp_r2]

            disp_occ_r1 = _adaptive_disocc_detection_disp(disp_l1)
            disp_occ_r2 = _adaptive_disocc_detection_disp(disp_l2)

            disp_occs_1 = [disp_occ_l1, disp_occ_r1]
            disp_occs_2 = [disp_occ_l2, disp_occ_r2]

            k_1_augs = [k_l1_aug, k_r1_aug]
            k_2_augs = [k_l2_aug, k_r2_aug]

            loss_cons, loss_cross, loss_ego, loss_static, loss_cm, loss_mask = self.consistency_loss(sfs_f, sfs_b,
                                                                                                     cms_f, cms_b,
                                                                                                     disps_1, disps_2,
                                                                                                     disp_occs_1, disp_occs_2,
                                                                                                     imgs_1_aug, imgs_2_aug,
                                                                                                     rigidity_mask_f, rigidity_mask_b,
                                                                                                     k_1_augs, k_2_augs, cam_r2l,
                                                                                                     aug_size, baseline)

            loss_cons_sum += loss_cons * self._weights[ii]
            loss_cm_sum += loss_cm
            loss_cross_sum += loss_cross 
            loss_ego_sum += loss_ego
            loss_mask_sum += loss_mask
            loss_static_sum += loss_static

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        c_loss= loss_cons_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        max_val = torch.max(max_val, c_loss)

        f_weight = max_val / f_loss
        d_weight = max_val / d_loss
        c_weight = max_val / c_loss

        if self._args.debugging:
          print(f"flow loss: {f_weight}, {f_loss} \n disp loss: {d_weight}, {d_loss} \n cons loss: {c_weight}, {c_loss}")

        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight + loss_cons_sum

        loss_cons_dict = {}
        loss_cons_dict['cross'] = loss_cross_sum
        loss_cons_dict['ego'] = loss_ego_sum
        loss_cons_dict['static'] = loss_static_sum
        loss_cons_dict['cm'] = loss_cm_sum
        loss_cons_dict['mask'] = loss_mask_sum

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["cons"] = loss_cons_sum
        loss_dict["cons_dict"] = loss_cons_dict
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict

