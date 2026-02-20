"""
model.py - MobileNetV3-Large via timm for facial expression recognition.
"""

import timm


def build_model(cfg: dict):
    """
    Create a MobileNetV3-Large model with a custom classifier head.

    timm.create_model automatically replaces the final classifier layer
    to output `num_classes` logits.
    """
    model_cfg = cfg["model"]
    model = timm.create_model(
        model_cfg["name"],
        pretrained=model_cfg["pretrained"],
        num_classes=model_cfg["num_classes"],
    )
    return model
