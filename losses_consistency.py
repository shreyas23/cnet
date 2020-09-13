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


class Loss_SceneFlow_SelfSup_Pose(nn.Module):
  def __init__(self, args):
    super(Loss_SceneFlow_SelfSup_Pose, self).__init__()
    self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
    self._ssim_w = 0.85
    self._disp_smooth_w = 0.1
    self._sf_3d_pts = 0.2
    self._sf_3d_sm = 200

    self._cam_ph_w = 1.0
    

  def pose_loss(self, 
                pose_f, pose_b, 
                disp_l1, disp_l2, 
                disp_occ_l1, disp_occ_l2,
                k_l1, k_l2,
                img_l1, img_l2,
                aug_size, ii):

    _, _, _, w = disp_l1.size()

    disp_l1 = disp_l1 * w
    disp_l2 = disp_l2 * w
    
    depth_l1 = _disp2depth_kitti_K(disp_l1, k_l1[:, 0, 0])
    depth_l2 = _disp2depth_kitti_K(disp_l2, k_l2[:, 0, 0])

    flow_f = pose2flow(depth_l1.squeeze(dim=0), pose_f, k_l1, torch.inverse(k_l1))
    flow_b = pose2flow(depth_l2.squeeze(dim=0), pose_b, k_l2, torch.inverse(k_l2))
    occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
    occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

    img_l2_warp = flow_warp(img_l2, flow_f)
    img_l1_warp = flow_warp(img_l1, flow_b)

    flow_f_diff = _reconstruction_error(img_l1, img_l2_warp, self._ssim_w)
    flow_b_diff = _reconstruction_error(img_l2, img_l1_warp, self._ssim_w)

    flow_f_loss = flow_f_diff[occ_map_f].mean()
    flow_b_loss = flow_b_diff[occ_map_b].mean()

    flow_f_diff[~occ_map_f].detach_()
    flow_b_diff[~occ_map_b].detach_()

    loss_ph = (flow_f_loss + flow_b_loss) * self._cam_ph_w

    return loss_ph

  def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

    img_r_warp = _generate_image_left(img_r_aug, disp_l)
    left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

    # Photometric loss
    img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) +
                _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)
    loss_img = (img_diff[left_occ]).mean()
    img_diff[~left_occ].detach_()

    # Disparities smoothness
    loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

    return loss_img + self._disp_smooth_w * loss_smooth, left_occ

  def sceneflow_loss(self,
                      sf_f, sf_b,
                      disp_l1, disp_l2,
                      disp_occ_l1, disp_occ_l2,
                      k_l1_aug, k_l2_aug,
                      img_l1_aug, img_l2_aug,
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

    # Loss Summnation
    sceneflow_loss = loss_im + self._sf_3d_pts * \
        loss_pts + self._sf_3d_sm * loss_3d_s

    return sceneflow_loss, loss_im, loss_pts, loss_3d_s

  def detaching_grad_of_outputs(self, output_dict):

    for ii in range(0, len(output_dict['flow_f'])):
      output_dict['flow_f'][ii].detach_()
      output_dict['flow_b'][ii].detach_()
      output_dict['poses_f'][ii].detach_()
      output_dict['poses_b'][ii].detach_()
      output_dict['disp_l1'][ii].detach_()
      output_dict['disp_l2'][ii].detach_()
    return

  def forward(self, output_dict, target_dict):

    loss_dict = {}

    # batch_size = target_dict['input_l1'].size(0)
    loss_sf_sum = 0
    loss_po_sum = 0
    loss_dp_sum = 0
    loss_sf_2d = 0
    loss_sf_3d = 0
    loss_sf_sm = 0

    k_l1_aug = target_dict['input_k_l1_aug']
    k_l2_aug = target_dict['input_k_l2_aug']
    aug_size = target_dict['aug_size']

    disp_r1_dict = output_dict['output_dict_r']['disp_l1']
    disp_r2_dict = output_dict['output_dict_r']['disp_l2']

    for ii, (sf_f, sf_b, 
             pose_f, pose_b, 
             disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'],
                                                                  output_dict['poses_f'], output_dict['poses_b'],
                                                                  output_dict['disp_l1'], output_dict['disp_l2'],
                                                                  disp_r1_dict, disp_r2_dict)):

      assert(sf_f.size()[2:4] == sf_b.size()[2:4])
      assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
      assert(sf_f.size()[2:4] == disp_l2.size()[2:4])

      # For image reconstruction loss
      img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
      img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
      img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
      img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

      # Disp Loss
      loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(
          disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
      loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(
          disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
      loss_dp_sum = loss_dp_sum + \
          (loss_disp_l1 + loss_disp_l2) * self._weights[ii]

      # Sceneflow Loss
      loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b,
                                                                         disp_l1, disp_l2,
                                                                         disp_occ_l1, disp_occ_l2,
                                                                         k_l1_aug, k_l2_aug,
                                                                         img_l1_aug, img_l2_aug,
                                                                         aug_size, ii)

      loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]
      loss_sf_2d = loss_sf_2d + loss_im
      loss_sf_3d = loss_sf_3d + loss_pts
      loss_sf_sm = loss_sf_sm + loss_3d_s

      loss_pose = self.pose_loss(pose_f, pose_b, 
                                 disp_l1, disp_l2, 
                                 disp_occ_l1, disp_occ_l2, 
                                 k_l1_aug, k_l2_aug, 
                                 img_l1_aug, img_l2_aug, 
                                 aug_size, ii)

      loss_po_sum = loss_po_sum + loss_pose * self._weights[ii]

    # finding weight
    f_loss = loss_sf_sum.detach()
    d_loss = loss_dp_sum.detach()
    p_loss = loss_po_sum.detach()

    max_val = torch.max(f_loss, d_loss)
    max_val = torch.max(max_val, p_loss)

    f_weight = max_val / f_loss
    d_weight = max_val / d_loss
    p_weight = max_val / p_loss

    total_loss = (loss_sf_sum * f_weight) + (loss_dp_sum * d_weight) + (loss_po_sum * p_weight)

    loss_dict = {}
    loss_dict["dp"] = loss_dp_sum
    loss_dict["sf"] = loss_sf_sum
    loss_dict["po"] = loss_po_sum
    loss_dict["s_2"] = loss_sf_2d
    loss_dict["s_3"] = loss_sf_3d
    loss_dict["s_3s"] = loss_sf_sm
    loss_dict["total_loss"] = total_loss

    self.detaching_grad_of_outputs(output_dict['output_dict_r'])

    return loss_dict


class Loss_SceneFlow_SelfSup_PoseMask(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_Consistency, self).__init__()

        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_smooth_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200

        self._cam_ph_w = 1.0
        self._mask_w = 1.0
    
    def mask_loss(self):
      return 0
      
    def camflow_loss(self, 
                     pose_f, pose_b, 
                     disp_l1, disp_l2, 
                     disp_occ_l1, disp_occ_l2,
                     k_l1, k_l2,
                     img_l1, img_l2,
                     aug_size, ii):

      _, _, _, w = disp1.size()

      disp_l1 *= w
      disp_l2 *= w
      
      depth1 = _disp2depth_kitti_K(disp1, intrinsics[:, 0, 0])
      depth2 = _disp2depth_kitti_K(disp2, intrinsics[:, 0, 0])

      flow_f = pose2flow(depth1, pose_f, k_l1, torch.inverse(k_l1))
      flow_b = pose2flow(depth2, pose_b, k_l2, torch.inverse(k_l2))
      occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
      occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

      img_l2_warp = flow_warp(img_l2, flow_f)
      img_l1_warp = flow_warp(img_l1, flow_b)

      flow_f_diff = _reconstruction_error(img_l1, img_l2_warp, self._ssim_w)
      flow_b_diff = _reconstruction_error(img_l2, img_l1_warp, self._ssim_w)

      flow_f_loss = flow_f_diff[occ_map_f].mean()
      flow_b_loss = flow_b_diff[occ_map_b].mean()

      flow_f_diff[~occ_map_f].detach_()
      flow_b_diff[~occ_map_b].detach_()

      loss_ph = (flow_f_loss + flow_b_loss) * self.cam_ph_w

      return loss_ph

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
                       k_l1_aug, k_l2_aug,
                       img_l1_aug, img_l2_aug,
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

        # Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_pts * \
            loss_pts + self._sf_3d_sm * loss_3d_s

        return sceneflow_loss, loss_im, loss_pts, loss_3d_s

    def detaching_grad_of_outputs(self, output_dict):

        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        # batch_size = target_dict['input_l1'].size(0)
        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        loss_sf_3d = 0
        loss_sf_sm = 0

        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'],
                                                                                  output_dict['flow_b'],
                                                                                  output_dict['disp_l1'],
                                                                                  output_dict['disp_l2'],
                                                                                  disp_r1_dict,
                                                                                  disp_r2_dict)):

            assert(sf_f.size()[2:4] == sf_b.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l2.size()[2:4])

            # For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            # Disp Loss
            loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(
                disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(
                disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + \
                (loss_disp_l1 + loss_disp_l2) * self._weights[ii]

            # Sceneflow Loss
            loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b,
                                                                               disp_l1, disp_l2,
                                                                               disp_occ_l1, disp_occ_l2,
                                                                               k_l1_aug, k_l2_aug,
                                                                               img_l1_aug, img_l2_aug,
                                                                               aug_size, ii)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]
            loss_sf_2d = loss_sf_2d + loss_im
            loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss

        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict


class Loss_SceneFlow_SelfSup_PoseMaskConsistency(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_Consistency, self).__init__()

        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_smooth_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200

        self._cam_ph_w = 1.0
        self._mask_w = 1.0
        """
        static reconstruction
        dynamic reconstruction
        left/right pose consistency
        camera motion/sf static scene consistency
        semantic/motion consistency ??? 
        """
        self._cons_weights = [1.0, 1.0, 1.0, 1.0]
    
    def camflow_loss(self, 
                     pose_f, pose_b, 
                     disp_l1, disp_l2, 
                     disp_occ_l1, disp_occ_l2,
                     k_l1, k_l2,
                     img_l1, img_l2,
                     aug_size, ii):

      _, _, _, w = disp1.size()

      disp_l1 *= w
      disp_l2 *= w
      
      depth1 = _disp2depth_kitti_K(disp1, intrinsics[:, 0, 0])
      depth2 = _disp2depth_kitti_K(disp2, intrinsics[:, 0, 0])

      flow_f = pose2flow(depth1, pose_f, k_l1, torch.inverse(k_l1))
      flow_b = pose2flow(depth2, pose_b, k_l2, torch.inverse(k_l2))
      occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
      occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

      img_l2_warp = flow_warp(img_l2, flow_f)
      img_l1_warp = flow_warp(img_l1, flow_b)

      flow_f_diff = _reconstruction_error(img_l1, img_l2_warp, self._ssim_w)
      flow_b_diff = _reconstruction_error(img_l2, img_l1_warp, self._ssim_w)

      flow_f_loss = flow_f_diff[occ_map_f].mean()
      flow_b_loss = flow_b_diff[occ_map_b].mean()

      flow_f_diff[~occ_map_f].detach_()
      flow_b_diff[~occ_map_b].detach_()

      loss_ph = (flow_f_loss + flow_b_loss) * self.cam_ph_w

      return loss_ph

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
                       k_l1_aug, k_l2_aug,
                       img_l1_aug, img_l2_aug,
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

        # Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_pts * \
            loss_pts + self._sf_3d_sm * loss_3d_s

        return sceneflow_loss, loss_im, loss_pts, loss_3d_s

    def detaching_grad_of_outputs(self, output_dict):

        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        # batch_size = target_dict['input_l1'].size(0)
        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        loss_sf_3d = 0
        loss_sf_sm = 0

        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'],
                                                                                  output_dict['flow_b'],
                                                                                  output_dict['disp_l1'],
                                                                                  output_dict['disp_l2'],
                                                                                  disp_r1_dict,
                                                                                  disp_r2_dict)):

            assert(sf_f.size()[2:4] == sf_b.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l2.size()[2:4])

            # For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            # Disp Loss
            loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(
                disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(
                disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + \
                (loss_disp_l1 + loss_disp_l2) * self._weights[ii]

            # Sceneflow Loss
            loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b,
                                                                               disp_l1, disp_l2,
                                                                               disp_occ_l1, disp_occ_l2,
                                                                               k_l1_aug, k_l2_aug,
                                                                               img_l1_aug, img_l2_aug,
                                                                               aug_size, ii)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]
            loss_sf_2d = loss_sf_2d + loss_im
            loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss

        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict
