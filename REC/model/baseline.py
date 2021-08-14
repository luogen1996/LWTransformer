import torch
import torch.nn as nn
import json
from config.base_config import cfg
import torch.nn.functional as F
from model.adapter import Adapter
class baseNet(nn.Module):
    def __init__(self, vocab_size, pretrained_emb=None):
        super(baseNet, self).__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=cfg.WORD_EMB_SIZE
        )
        self.input_dropout = nn.Dropout(0.1)
        self.mlp = nn.Sequential(nn.Linear(cfg.WORD_EMB_SIZE, cfg.WORD_EMB_SIZE),
                                 nn.ReLU())
        # Loading the GloVe embedding weights
        # if cfg.USE_GLOVE:
        #     self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.adapter=Adapter()

        self.lstm = nn.LSTM(
            input_size=cfg.WORD_EMB_SIZE,
            hidden_size=cfg.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.fuse_proj=nn.Linear(3072,cfg.HIDDEN_SIZE)
        self.conf_proj = nn.Linear(cfg.HIDDEN_SIZE , 1)
        self.bbox_proj = nn.Linear(cfg.HIDDEN_SIZE , 4)


    def forward(self,*inputs):

        qvec, img_feat, spt_feat = inputs
        # input_lengths = (qvec != 0).sum(1)

        lang_feat = self.embedding(qvec)
        lang_feat=self.input_dropout(lang_feat)
        lang_feat=self.mlp(lang_feat)
        lang_feat, (h_n, c_n) = self.lstm(lang_feat)
        # print(qvec)

        lang_pooled=h_n[-1]
        # L2 Normalize
        lang_feat = F.normalize(input=lang_pooled,dim=-1, p=2)  # perform Lp normalization of inputs over specified dimension,这里p=2
        img_feat= F.normalize(input=img_feat,dim=-1, p=2)


        # vis_feat,vis_mask=self.adapter(img_feat,spt_feat)

        #fusion
        lang_feat=lang_feat.unsqueeze(1).repeat(1,img_feat.size(1),1)
        fuse_feat=torch.cat([lang_feat,img_feat],-1)
        fuse_feat=F.relu(self.fuse_proj(fuse_feat))

        # # predict confidence
        query_score_pred = self.conf_proj(fuse_feat).view(-1, cfg.RPN_TOPN)

        # # predict bbox
        query_bbox_pred = self.bbox_proj(fuse_feat)

        return query_score_pred, query_bbox_pred


