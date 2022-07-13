import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Non-distributed generalized matrix multiply"""

    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim * 4)
        # Output dimension is the same as input dimension since all GEMM are
        # all-reduced, not gathered
        self.fc2 = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        self.nonlin = F.gelu

    def forward(self, x):
        xa = self.fc1(x)
        y = self.nonlin(xa)
        z = self.fc2(y)
        return z

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SelfAttention(nn.Module):
    """Non-distributed self attention module"""

    def __init__(self, input_data_dim, hidden_dim, msa_heads, qkv_dropout, msa_dropout):
        super(SelfAttention, self).__init__()
        self.input_data_dim = input_data_dim
        self.hidden_dim = hidden_dim

        self.msa_heads = msa_heads

        # hidden dim for this layer is 2 * model dim
        self.ln = nn.LayerNorm(input_data_dim)
        self.fc1 = nn.Linear(input_data_dim, hidden_dim * 3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.qkv_dropout = nn.Dropout(p=qkv_dropout)
        self.msa_dropout = nn.Dropout(p=msa_dropout)

    def forward(self, x):
        B, S, E = x.shape
        x = self.ln(x)
        qkv = self.fc1(x)
        qkv_heads = qkv.reshape(self.msa_heads, B, S, -1)
        q, k, v = torch.split(qkv_heads, qkv_heads.shape[-1] // 3, dim=-1)
        qk_heads = (
            torch.einsum("hbse,hbSe->hbsS", q, k) * (self.input_data_dim * 2) ** -0.5
        )
        att_heads = F.softmax(qk_heads, dim=-1)
        att_heads = self.qkv_dropout(att_heads)
        full_att = torch.einsum("hbsS,hbSe->hbse", att_heads, v)
        full_att = full_att.reshape(B, S, -1)
        out = self.fc2(full_att)
        out = self.msa_dropout(out)
        return out

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
