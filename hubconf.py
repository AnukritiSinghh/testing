import torch

from models.backbone import Backbone, Joiner
from models.detr import DETR, PostProcess
from models.position_encoding import PositionEmbeddingSine
from models.segmentation import DETRsegm, PostProcessPanoptic
from models.transformer import Transformer

dependencies = ["torch", "torchvision"]


def _make_detr(backbone_name: str, dilation=False, num_classes=91, mask=False):
    hidden_dim = 256
    backbone = Backbone(backbone_name, train_backbone=True, return_interm_layers=mask, dilation=dilation)
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels
    transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=True)
    detr = DETR(backbone_with_pos_enc, transformer, num_classes=num_classes, num_queries=100)
    if mask:
        return DETRsegm(detr)
    return detr

def detr_resnet101_offroad(
    pretrained=False, num_classes=250, threshold=0.85, return_postprocessor=False
):
    model = _make_detr("resnet101", dilation=False, num_classes=num_classes, mask=True)
    is_thing_map = {i: i <= 90 for i in range(250)}
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/AnukritiSinghh/checkpoints/blob/master/checkpoint_resume.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcessPanoptic(is_thing_map, threshold=threshold)
    return modell
