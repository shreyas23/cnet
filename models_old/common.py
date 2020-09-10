import torch
import torch.nn as nn
import torch.nn.function as tf

""" some functions borrowed and modified from SENSE
    https://github.com/NVlabs/SENSE/blob/master/sense/models/common.py
"""
def make_bn_layer(bn_type, plane):
    if bn_type == 'plain':
        return nn.BatchNorm2d(plane)
    else:
        raise Exception('Not supported BN type: {}.'.format(bn_type))

def convbn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, bn_type='plain', use_relu=True):
    layers = []
    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                  stride=stride, padding=padding, dilation=dilation, bias=bias))   
    layers.append(make_bn_layer(bn_type, out_planes))
    if use_relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

    
def flow_warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())
    
    mask[mask.data < 0.9999] = 0
    mask[mask.data > 0] = 1
    
    return output*mask

def disp_warp(rim, disp):
    """
    warp stereo image (right image) with disparity
    rim: [B, C, H, W] image/tensor
    disp: [B, 1, H, W] (left) disparity
    for disparity (left), we have
        left_image(x,y) = right_image(x-d(x,y),y)
    """
    B, C, H, W = rim.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if rim.is_cuda:
        grid = grid.cuda()
    vgrid = grid

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*(vgrid[:,0,:,:]-disp[:,0,:,:])/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    return nn.functional.grid_sample(rim, vgrid.permute(0,2,3,1))