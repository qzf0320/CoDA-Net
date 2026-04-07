import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv

from Frequency_aware import FreqAwareGraphBlock
from GCT import GraphCTransformerLayer


class GraphTransformerDenoiser(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, time_dim = 0, cond_dim=0):
        super(GraphTransformerDenoiser, self).__init__()

        if hidden_dim is None:
            hidden_dim = [64, 128, 256, 512, 1024]

        self.cond_dim = cond_dim
        self.cond0_proj = nn.Linear(in_dim * 2, in_dim)
        self.cond1_proj = nn.Linear(in_dim + 2, in_dim)
        self.time_proj = nn.Linear(in_dim * 2, hidden_dim[0])
        self.cond_proj = nn.Linear(in_dim * 2 + 2, in_dim)

        self.input_proj = nn.Sequential(
                            nn.Linear(in_dim*2+2, hidden_dim[0]),
                            nn.ReLU(),
                            nn.BatchNorm1d(hidden_dim[0]))

        self.layer1 = GraphCTransformerLayer(hidden_dim[0], hidden_dim[1], heads=8)
        self.layer2 = GraphCTransformerLayer(hidden_dim[1], hidden_dim[2], heads=4)
        self.layer3 = GraphCTransformerLayer(hidden_dim[2], hidden_dim[3], heads=2)
        self.layer4 = GraphCTransformerLayer(hidden_dim[3], hidden_dim[4], heads=1)

        self.freq1 = FreqAwareGraphBlock(hidden_dim[1], [in_dim, cond_dim], time_dim, hidden_dim[1], 2)
        self.freq2 = FreqAwareGraphBlock(hidden_dim[2], [in_dim, cond_dim], time_dim, hidden_dim[2], 2)
        self.freq3 = FreqAwareGraphBlock(hidden_dim[3], [in_dim, cond_dim], time_dim, hidden_dim[3], 2)
        self.freq4 = FreqAwareGraphBlock(hidden_dim[4], [in_dim, cond_dim], time_dim, hidden_dim[4], 2)

        self.time_embed = nn.Linear(1, in_dim)
        self.dropout = nn.Dropout(0.5)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim[-1], hidden_dim[-1]),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim[-1], out_dim or in_dim)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, TransformerConv):
                nn.init.xavier_uniform_(m.lin_query.weight)
                nn.init.xavier_uniform_(m.lin_key.weight)
                nn.init.xavier_uniform_(m.lin_value.weight)

            elif isinstance(m, GCNConv):
                nn.init.xavier_uniform_(m.lin.weight)

    def forward(self, x, edge_index, t = None, cond=None):
        cond1 = cond[1].repeat_interleave(116, dim=0)
        cond1 = cond1.float()
        t_emb = self.time_embed(t.view(-1, 1).float())
        x = torch.cat([x, cond[0], cond1], dim=1)
        x = self.cond_proj(x)
        x = torch.cat([x, t_emb], dim=1)
        x = self.time_proj(x)

        x = self.layer1(x, edge_index)
        x = self.freq1(x, edge_index, cond[0], cond1, t_emb)
        x = self.layer2(x, edge_index)
        x = self.freq2(x, edge_index, cond[0], cond1, t_emb)
        x = self.layer3(x, edge_index)
        x = self.freq3(x, edge_index, cond[0], cond1, t_emb)
        x = self.layer4(x, edge_index)
        x = self.freq4(x, edge_index, cond[0], cond1, t_emb)

        return self.output_proj(x)


if __name__ == '__main__':
    x = torch.randn([10, 116, 110])
    c = torch.zeros(x.shape[0]).long()
    c = F.one_hot(c, 2)
    t = torch.randint(0, 100, (10,)).long()
    # model = TransformerDenoiser(110, 116, 112, head=4)
    # print(model(x, c, t).shape)

