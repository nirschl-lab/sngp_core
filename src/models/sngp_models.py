import torch
import torch.nn as nn
from typing import Iterable, Optional
from src.models.sngp.sngp_classification_layer import SNGP
from src.models.sngp.gaussian_process import RandomFeatureGaussianProcess
from torch.nn.utils import spectral_norm
from src.utils.helper import count_num_parms

#205441
class SNGPModel(nn.Module):
    def __init__(self, in_dim=384, num_classes=8, p_drop: float = 0.1, ):
        super().__init__()
        reduction_dim = 256
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p_drop)
        
        self.pre_classifier = nn.utils.spectral_norm(
            nn.Linear(in_dim, reduction_dim))

        self.classifier = SNGP(
                in_features=reduction_dim,
                num_classes=num_classes,
                kernel_scale_trainable=True,
                scale_random_features=True,
                normalize_input=False,
                covariance_momentum=0.999,
                return_dict=False,
            ) # H -> H -> R -> D (R=128 D=1024) R < H, D >> R

    def forward(self, x):
        x = self.dropout(self.activation(self.pre_classifier(x)))
        return self.classifier(x)

#205441
class SNGPCustom(nn.Module):
    def __init__(self, in_dim=384, num_classes=8, hidden: Iterable[int] = (256, 512,), p_drop: float = 0.1):
        super().__init__()

        layers = []
        last = in_dim
        for h in hidden:
            layers += [
                spectral_norm(nn.Linear(last, h)),  # apply spectral norm here
                nn.ReLU(inplace=True),
                nn.Dropout(p_drop)
            ]
            last = h
        self.SN_backbone = nn.Sequential(*layers)

        in_features = hidden[-1]
        reduction_dim = 128
        self.reduce_dim_layer = nn.utils.spectral_norm(
            nn.Linear(in_features, reduction_dim)
        )
        
        # random_features >> gp_in_features
        gp_in_features = reduction_dim #hidden[-1]
        
        gp_kwargs = {
            "in_features":gp_in_features,
            "out_features":num_classes,
            "random_features": 1024,
            "scale_random_features":True,
            "normalize_input":False,
            "kernel_scale_trainable": True,
            "covariance_momentum": 0.999,
            "covariance_likelihood": "gaussian",
            "return_dict": False,
            "output_bias_trainable": False,
        }

        self.gp_classifier = RandomFeatureGaussianProcess(**gp_kwargs)

    def forward(self, x):
        x = self.SN_backbone(x)
        x = self.reduce_dim_layer(x)
        return self.gp_classifier(x)
if __name__ == '__main__':
    sngp_model = SNGPModel(in_dim=384, num_classes=8)
    print('SNGP Model - ' + count_num_parms(sngp_model))
    sngp_custom_model = SNGPCustom(in_dim=384, num_classes=8)
    print('SNGP Custom Model - ' + count_num_parms(sngp_custom_model))