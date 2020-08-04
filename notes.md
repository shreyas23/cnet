# Scene Flow Estimation Notes

## Camera Motion Estimation

- More views the better for camera motion/egomotion estimation

- 

## Overall Architecture

### Shared Encoder + Shared Split Disparity/SceneFlow + Camera Motion + Motion Segmentation Mask Decoders

    Main Issue with this is chicken/egg optimization problem with the motion segmentation mask and split scene flow estimators.  
    We use the motion segmentation mask to merge the static + dynamic scene flows and disparities

## Loss Functions

### Consistency

### General 