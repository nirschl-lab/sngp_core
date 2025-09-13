#!/usr/bin/env python3
"""model_registry.py in src/sngp_core/models.

Adapted from:
https://github.com/sanketx/AL-foundation-models/blob/main/ALFM/src/models/registry.py
"""

from enum import Enum


class ModelType(Enum):
    """Enum of supported Models."""

    # General models trained on natural images
    # ViT
    openai_vit_B16 = ("ViT-B-16", "openai")

    # contrastive models
    openclip_vit_B16 = ("ViT-B-16", "laion2b_s34b_b88k")
    openclip_vit_L14 = ("ViT-L-14", "laion2b_s32b_b82k")
    openclip_vit_H14 = ("ViT-H-14", "laion2b_s32b_b79k")
    openclip_vit_g14 = ("ViT-g-14", "laion2b_s34b_b88k")
    openclip_vit_G14 = ("ViT-bigG-14", "laion2b_s39b_b160k")

    # DINOv2
    dinov2_vit_S14 = ("facebookresearch/dinov2", "dinov2_vits14")
    dinov2_vit_B14 = ("facebookresearch/dinov2", "dinov2_vitb14")
    dinov2_vit_L14 = ("facebookresearch/dinov2", "dinov2_vitl14")
    dinov2_vit_g14 = ("facebookresearch/dinov2", "dinov2_vitg14")

    # DINOv2 with registers
    dinov2_vit_S14_reg = ("facebookresearch/dinov2", "dinov2_vits14_reg")
    dinov2_vit_B14_reg = ("facebookresearch/dinov2", "dinov2_vitb14_reg")
    dinov2_vit_L14_reg = ("facebookresearch/dinov2", "dinov2_vitl14_reg")
    dinov2_vit_G14_reg = ("facebookresearch/dinov2", "dinov2_vitg14_reg")

    # DINOv3
    dinov3_vit_S16 = "facebook/dinov3-vits16-pretrain-lvd1689m"  # 384
    dinov3_vit_S16plus = "facebook/dinov3-vits16plus-pretrain-lvd1689m"  # 384
    dinov3_vit_B16 = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # 768
    dinov3_vit_L16 = "facebook/dinov3-vitl16-pretrain-lvd1689m"  # 1024
    dinov3_vit_H16plus = "facebook/dinov3-vith16plus-pretrain-lvd1689m"  # 1280
    dinov3_vit_7B16 = "facebook/dinov3-vit7b16-pretrain-lvd1689m"  # 4096
    dinov3_convnext_S = "facebook/dinov3-convnext-small-pretrain-lvd1689m"

    # Domain specific models: Pathology
    uni_vit_L16 = ("ViT-L/16", "Uni")
    conch = ("conch", "conch")

    # Prov Gigapath
    provgigapath_dinov2_vit_G14 = ("prov-gigapath", "prov-gigapath")

    # Phikon
    phikon = ("owkin/phikon", "phikon")

    # Paige AI Virchow
    virchow_vit_H14 = ("paige-ai/Virchow", "virchow")

    # custom models
    amyloid_dino_vit_S16 = ("vit_small_patch16_224.dino", "amyloid_dino_vit_s16.pth")
