import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class FreqAwareGraphBlock(nn.Module):
    def __init__(self, F, cond_dim, time_dim, hid, heads=4):
        super().__init__()
        C1, C2 = cond_dim
        self.freq_mlp = nn.Linear((F // 2 + 1) * 2, hid)

        self.heads = heads
        assert hid % heads == 0, "hid 必须能被 heads 整除"
        self.dk = hid // heads

        self.to_q   = nn.Linear(hid, hid, bias=False)
        self.to_k1  = nn.Linear(hid, hid, bias=False)
        self.to_v1  = nn.Linear(hid, hid, bias=False)
        self.to_k2  = nn.Linear(hid, hid, bias=False)
        self.to_v2  = nn.Linear(hid, hid, bias=False)

        self.gate_mlp = nn.Sequential(
            nn.Linear(hid * 3, hid // 2),
            nn.ReLU(),
            nn.Linear(hid // 2, 2)
        )


        self.gcn = GCNConv(hid, hid)

        self.cond_mlp1 = nn.Linear(C1, hid)
        self.cond_mlp2 = nn.Linear(C2, hid)
        self.time_mlp  = nn.Linear(time_dim, hid)

        self.fuse = nn.Linear(hid * 3, hid)

    def forward(self, x, edge_index, cond1, cond2, t):
        M, D = x.shape

        Xf = torch.fft.rfft(x, dim=-1)            # complex64: (M, F//2+1)

        Zf = torch.cat([Xf.real, Xf.imag], dim=-1)

        h_freq = F.relu(self.freq_mlp(Zf))        # (M, hid)

        Q  = self.to_q(h_freq).view(M, self.heads, self.dk)    # (M, heads, dk)
        C1 = self.cond_mlp1(cond1)                             # (M, hid)
        C2 = self.cond_mlp2(cond2)                             # (M, hid)
        t_emb = self.time_mlp(t)
        K1 = self.to_k1(C1).view(M, self.heads, self.dk)       # (M, heads, dk)
        V1 = self.to_v1(C1).view(M, self.heads, self.dk)       # (M, heads, dk)
        K2 = self.to_k2(C2).view(M, self.heads, self.dk)       # (M, heads, dk)
        V2 = self.to_v2(C2).view(M, self.heads, self.dk)       # (M, heads, dk)

        logits1 = (Q * K1).sum(dim=-1) / (self.dk ** 0.5)  # (M, heads)
        α1 = torch.softmax(logits1, dim=-1).unsqueeze(-1)  # (M, heads, 1)
        h_att1 = (α1 * V1).reshape(M, -1)  # (M, hid)

        logits2 = (Q * K2).sum(dim=-1) / (self.dk ** 0.5)
        α2 = torch.softmax(logits2, dim=-1).unsqueeze(-1)
        h_att2 = (α2 * V2).reshape(M, -1)

        gate_in = torch.cat([h_freq, C1, C2], dim=-1)                 # (M, hid + C1)
        gate_logits = self.gate_mlp(gate_in)                       # (M, 2)
        gate_coeff = F.softmax(gate_logits, dim=-1)                # (M, 2)

        pi1 = gate_coeff[:, 0].unsqueeze(-1)                       # (M, 1)
        pi2 = gate_coeff[:, 1].unsqueeze(-1)                       # (M, 1)

        h_att = pi1 * h_att1 + pi2 * h_att2                        # (M, hid)

        h_gcn = F.relu(self.gcn(h_att, edge_index))                # (M, hid)

        h_fuse = torch.cat([h_att, h_gcn, t_emb], dim=-1)          # (M, hid*3)
        out = F.relu(self.fuse(h_fuse))                            # (M, hid)

        return out