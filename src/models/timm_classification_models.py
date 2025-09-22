import timm
import torch
import torch.nn as nn
from src.models.sngp.sgnp_classification_layer import SNGP
from typing import Optional, Iterable

class TimmBasicClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, in_chans=3):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans
        )
    
    def forward(self, x):
        return self.backbone(x)

class TimmSNGPClassifier(nn.Module):
    def __init__(self, 
                 model_name, 
                 num_classes, 
                 pretrained=True, 
                 in_chans=3,
                 reduction_dim=512):
        super().__init__()

        self.backbone = timm.create_model(
                            model_name,
                            pretrained=pretrained,
                            )
        
        in_feats = getattr(self.backbone, "num_features", None)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce_dim = nn.Linear(in_feats, reduction_dim)

        self.sngp_classifier = SNGP(
                in_features=reduction_dim,
                num_classes=num_classes,
                kernel_scale_trainable=True,
                scale_random_features=True,
                normalize_input=False,
                covariance_momentum=0.999,
                return_dict=False,
            )
        
    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x).flatten(1)
        x = self.reduce_dim(x)
        return self.sngp_classifier(x)[0]


if __name__ == '__main__':
    batch_size = 2
    in_chans = 3
    height = width = 224
    x = torch.randn(batch_size, in_chans, height, width)
    model_name = "resnet50"
    num_classes = 8
    basic_classifier = TimmBasicClassifier(model_name, num_classes, pretrained=True, in_chans=3)
    sngp_model = TimmSNGPClassifier(model_name, 
                 num_classes, 
                 pretrained=True, 
                 in_chans=3,
                 reduction_dim=512)
    base_out = basic_classifier(x)
    sngp_out = sngp_model(x)

    print(f"Input shape: {x.shape}")
    print(f"Base Output shape: {base_out.shape}")
    print(f"SNGP Output shape: {sngp_out.shape}")