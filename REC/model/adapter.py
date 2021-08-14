# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import torch.nn as nn
import torch
from config.base_config import cfg
def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)

class Adapter(nn.Module):
    def __init__(self, ):
        super(Adapter, self).__init__()
        imgfeat_linear_size=cfg.SPT_FEAT_DIM+cfg.BOTTOMUP_FEAT_DIM-1
        self.frcn_linear = nn.Linear(imgfeat_linear_size, cfg.HIDDEN_SIZE)

    def forward(self, frcn_feat,bbox_feat):

        img_feat_mask = make_mask(frcn_feat)

        # bbox_feat = self.bbox_linear(bbox_feat)
        frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        img_feat = self.frcn_linear(frcn_feat)

        return img_feat, img_feat_mask




