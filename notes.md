# Scene Flow Estimation Notes

## General Notes

    General idea is that the scene flow can be understood better if there was a better understanding of motion decomposition.
    
    One way to decompose the scene is through motion segmenation, however, what do we use as a learning signal for motion segmentation?

    Why do we need motion segmentation?
        - Provides a learning signal for the network to differentiate between cues that apply to static vs dynamic scenes

## Avoiding Chicken-Egg Situation

    - In order to merge the sceneflow/disparity decoders to learn properly, we need the mask segmentation network to be pre-trained and then place an emphasis on 
    - We need some sort of supervisory signal to learn motion segmentation
    - Pretrain the mask segmentation network using CARLA? 
    - Can't pretrain the static/dynamic optical flow subnetworks because it won't learn the concept of static vs dynamic.
    -

## Camera Motion Estimation

    - More views the better for camera motion/egomotion estimation
    -  

## Overall Architecture

### Shared Encoder + Shared Split Disparity/SceneFlow + Camera Motion + Motion Segmentation Mask Decoders

    Main Issue with this is chicken/egg optimization problem with the motion segmentation mask and split scene flow estimators.  
    We use the motion segmentation mask to merge the static + dynamic scene flows and disparities

## Baseline Loss Functions

- Static reconstruction loss for both static sceneflow decoder and camera motion + disparity
- Dynamic scene reconstruction loss (traditional optical flow loss)
- 3d point euclidean loss
- supervised binary cross entropy loss for motion mask pretraining
- self supervised consistency loss for motion mask segmentation vs. dynamic/static sceneflow

### Consistency
