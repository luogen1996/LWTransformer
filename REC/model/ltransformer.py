

from model.layer_utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch, math
from config.base_config import cfg
from model.adapter import Adapter
class MHAtt(nn.Module):
    def __init__(self, d_model,dh,h,n=1,drop_r=0.1):
        super(MHAtt, self).__init__()
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, int(d_model*n))
        self.linear_q = nn.Linear(d_model, int(d_model*n))
        self.dh=dh
        self.h=h
        self.n=n
        self.d_model=d_model
        self.dropout = nn.Dropout(drop_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
        n_batches,
        -1,
        self.h,
        int(self.dh)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
        n_batches,
        -1,
        self.h,
        int(self.dh*self.n)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
        n_batches,
        -1,
        self.h,
        int(self.dh*self.n)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
        n_batches,
        -1,
        self.d_model
        )

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
        query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
#Lightweight FFN
class GFFN(nn.Module):
    def __init__(self, __C):
        super(GFFN, self).__init__()

        self.linear1 = nn.Linear(__C.HIDDEN_SIZE ,__C.FF_SIZE )
        self.linear2 = nn.Linear(__C.FF_SIZE//__C.GROUP, __C.HIDDEN_SIZE//__C.GROUP)
        self.act = nn.ReLU(inplace=True)
        self.groups=__C.GROUP
        self.hiddens=__C.FF_SIZE
        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, x):
        b,l,d=x.size()
        x=self.act(self.linear1(x))

        x=self.dropout(x)
        x = self.linear2(x.view(b, l, self.groups, -1)).view(b, l, d)
        return x

class GMHAtt(nn.Module):
    def __init__(self, __C):
        super(GMHAtt, self).__init__()
        self.__C = __C
        self.mha=MHAtt(__C.HIDDEN_SIZE//__C.GROUP,__C.MULTI_HEAD,__C.HIDDEN_SIZE//__C.MULTI_HEAD,__C.EXPAND,__C.DROPOUT_R)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.groups=__C.GROUP

    def forward(self, v, k, q, mask):
        if self.groups>1:
         #split
            q = torch.cat(torch.split(q, q.size(-1)//self.groups,
            dim=-1), 0)
            k = torch.cat(torch.split(k, k.size(-1)//self.groups,
            dim=-1), 0)
            v = torch.cat(torch.split(v, v.size(-1)//self.groups,
            dim=-1), 0)
            mask = mask.repeat(self.groups, 1, 1, 1)

        # transformation
        atted=self.mha(v,k,q,mask)

        if self.groups > 1:
            #concatenation
            atted = torch.cat(
            torch.split(atted,atted.size(0)//self.groups,dim=0)
            , -1)
        atted = self.linear_merge(atted)
        return atted


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = GMHAtt(__C)
        self.ffn = GFFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = GMHAtt(__C)
        self.mhatt2 = GMHAtt(__C)

        self.ffn = GFFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x
# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

    def forward(self, y, x, y_mask, x_mask):
        # Get encoder last hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)


        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)

        return y, x

class LTransformer(nn.Module):
    def __init__(self, vocab_size,pretrained_emb):
        super(LTransformer, self).__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=cfg.WORD_EMB_SIZE
        )
        # Loading the GloVe embedding weights
        if cfg.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=cfg.WORD_EMB_SIZE,
            hidden_size=cfg.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter()

        self.backbone = MCA_ED(cfg)

        # Flatten to vector
        self.attflat_lang = AttFlat(cfg)

        # Classification layers
        self.proj_norm = LayerNorm(cfg.FLAT_OUT_SIZE)
        self.fuse_proj=nn.Linear(cfg.FLAT_OUT_SIZE*2, cfg.FLAT_OUT_SIZE)
        self.conf_proj = nn.Linear(cfg.FLAT_OUT_SIZE, 1)
        self.bbox_proj = nn.Linear(cfg.FLAT_OUT_SIZE, 1)
        self.dropout=nn.Dropout(cfg.DROPOUT_R)



    def forward(self,*inputs):
        ques_ix, img_feat, spt_feat = inputs
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        img_feat, img_feat_mask=self.adapter(img_feat,spt_feat)
        # print(qvec)
        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )


        #fusion
        lang_feat=lang_feat.unsqueeze(1).repeat(1,img_feat.size(1),1)
        fuse_feat=torch.cat([lang_feat,img_feat],-1)
        fuse_feat=F.relu(self.fuse_proj(fuse_feat))
        fuse_feat=self.dropout(fuse_feat)

        # # predict confidence
        query_score_pred = self.conf_proj(fuse_feat).view(-1, cfg.RPN_TOPN)

        # # predict bbox
        query_bbox_pred = self.bbox_proj(fuse_feat)

        return query_score_pred, query_bbox_pred
