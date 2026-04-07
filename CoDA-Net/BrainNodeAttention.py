import torch.nn as nn
from GCT import GraphTransformerLayer
from torch_scatter import scatter_softmax


class NodeAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = GraphTransformerLayer(hidden_dim, 116, heads=2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, h_flat, edge_index, batch_index):
        B = batch_index.max().item() + 1
        h_proj = self.proj(h_flat, edge_index)  # (B*116, 116)

        e = h_proj.mean(dim=-1)
        alpha = scatter_softmax(e, batch_index)  # (B*116,)

        h_weighted = (alpha.unsqueeze(-1) * h_proj)  # (B*116, 116)
        alpha_map = alpha.view(B, 116)

        return alpha_map, h_weighted
