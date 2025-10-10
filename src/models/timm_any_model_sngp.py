import timm
import torch
import torch.nn as nn
from src.models.sngp.sngp_classification_layer import SNGP


class TimmSNGPClassifier(nn.Module):
    def __init__(self, 
                 model_name, 
                 num_classes, 
                 pretrained=True,
                 reduction_dim=512):
        super().__init__()

        is_vit = self._looks_like_vit(model_name)

        if is_vit:
            self.backbone = timm.create_model(
                                model_name,
                                pretrained=pretrained,
                                num_classes=0
                                )
        else:
            self.backbone = timm.create_model(
                                model_name,
                                pretrained=pretrained,
                                num_classes=0,       
                                global_pool='avg'
                            )
        
        in_feats = getattr(self.backbone, "num_features", None)
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
        x = self.backbone(x)
        x = self.reduce_dim(x)
        return self.sngp_classifier(x)[0]
    
    def _looks_like_vit(self, model_name: str) -> bool:
        # heuristic that works well with timm naming
        name = model_name.lower()
        return (
            "vit" in name
            or "deit" in name
            or "swin" in name
            or "maxvit" in name
            or "eva" in name
            or "levit" in name
        )


if __name__ == '__main__':
    batch_size = 2
    in_chans = 3
    height = width = 224
    x = torch.randn(batch_size, in_chans, height, width)
    for model_name in ["resnet50", 'efficientnet_b0','vit_small_patch16_224.dino', 'swin_s3_base_224']:
        num_classes = 8
        model = TimmSNGPClassifier(model_name, 
                            num_classes, 
                            pretrained=True, 
                            reduction_dim=512)
        out = model(x)
        print(f"Model {model_name} input - {x.shape} output - {out.shape}")
        