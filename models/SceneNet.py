import torch 
import torch.nn as nn
import torch.nn.functional as tf

from models.common import convbn
from models.modules_sceneflow import ResNetEncoder
    
class SceneNet(nn.Module):
    def __init__(self, args):
        super(SceneNet, self).__init__()
        
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 5
        
        if args.use_pwc_encoder:
            self.encoder_chs = [3, 32, 64, 96, 128, 192, 256]
            self.feature_pyramid_extractor = FeatureExtractor(self.encoder_chs)
        elif args.use_resnet_encoder:
            encoder_chs = [3, 32, 64, 128, 256]
            self.pyramid_encoder = ResNetEncoder(encoder_chs)
        else:
            raise NotImplementedError
        
        self.
        self.refinement_layers = nn.ModuleList()
        
        self.dim_corr = (self.search_range * 2 + 1) ** 2