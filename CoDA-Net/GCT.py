import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GINConv, GCNConv


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.heads = heads
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Transformer Conv
        self.conv = TransformerConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)


        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()


        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, edge_index):
        # --- Self-Attention ---
        residual = self.residual_proj(x)
        x_norm = self.norm1(x)
        x_attn = self.conv(x_norm, edge_index)
        x = residual + self.res_scale * x_attn

        # --- FFN ---
        x_ffn = self.norm2(x)
        x_ffn = self.ffn(x_ffn)
        x = x + self.res_scale * x_ffn

        return x


class GraphCTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, dropout=0.1):
        super(GraphCTransformerLayer, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.trans = TransformerConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(out_dim)
        self.conv = GINConv(
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(out_dim, out_dim)
            )
        )
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.res_scale = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        x_norm1 = self.norm1(x)
        x_trans = self.trans(x_norm1, edge_index)
        x_proj = self.residual_proj(x)
        out1 = x_proj + self.res_scale * x_trans

        x_conv = self.conv(x, edge_index)
        x_norm2 = self.norm2(F.gelu(out1))
        h = self.alpha * x_norm2 + self.beta * x_conv + (1.0 - self.alpha - self.beta) * out1
        out = self.ffn(h)

        return out


