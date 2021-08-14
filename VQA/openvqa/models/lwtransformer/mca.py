# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math
import random
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

        self.layerdrop = __C.LAYERDROP_R
    def forward(self, y, x, y_mask, x_mask):
        # Get encoder last hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)


        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)

        return y, x
