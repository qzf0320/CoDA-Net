import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, TransformerConv, GINConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse

from GCT import GraphTransformerLayer


def create_batch(x, adj):
    data_list = []
    for i in range(x.size(0)):
        edge_index, _ = dense_to_sparse(adj[i])
        # edge_index = build_edge_index(x[i])
        data = Data(x=x[i], edge_index=edge_index)
        data_list.append(data)
    return Batch.from_data_list(data_list)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(out_dim)
        self.res_scale = nn.Parameter(torch.tensor(0.1))
        self.dropout_num = dropout

    def forward(self, x, edge_index):
        residual = x
        x = self.conv(x, edge_index)
        x = self.norm(x)
        if self.dropout_num == 0:
            return F.relu(x)
        else:
            return F.relu(self.dropout(x))


class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super(GINLayer, self).__init__()

        self.gin_layer = GINConv(
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU()
            )
        )

    def forward(self, x, edge_index):

        return self.gin_layer(x, edge_index)



class BasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads=2, dropout=0.1):
        super().__init__()
        self.attn = TransformerConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
        self.gcn  = GCNConv(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # x: (B*N, dim)
        h1 = self.attn(x, edge_index)
        h2 = self.gcn(x, edge_index)
        out = x + h1 + h2
        out = self.norm(out)
        return self.drop(F.relu(out))


class GraphTransformerExtractor(nn.Module):
    def __init__(self, in_dim, hidden_dim=None):
        super(GraphTransformerExtractor, self).__init__()

        if hidden_dim is None:
            hidden_dim = [32, 64, 128, 256, 512]

        self.input_proj = nn.Linear(in_dim, hidden_dim[0])

        self.layer1 = GraphTransformerLayer(hidden_dim[0], hidden_dim[1], heads=4)
        self.layer2 = GraphTransformerLayer(hidden_dim[1], hidden_dim[2], heads=4)
        self.layer3 = GraphTransformerLayer(hidden_dim[2], hidden_dim[3], heads=2)
        self.layer4 = GraphTransformerLayer(hidden_dim[3], hidden_dim[4], heads=1)

        self.output_proj = nn.Linear(hidden_dim[-1], in_dim)
        self.fc_mu = nn.Linear(hidden_dim[-1], in_dim)  # 计算均值
        self.fc_logvar = nn.Linear(hidden_dim[-1], in_dim)  # 计算对数方差
        self.dropout = nn.Dropout(0.5)
        # ----------- 参数初始化 -----------
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index, batch=None):
        if len(x.shape) == 3:
            batch = create_batch(x, edge_index)
            x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        x = self.input_proj(x)
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)
        x = self.layer4(x, edge_index)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        out = self.dropout(self.reparameterize(mu, logvar))

        return out, global_mean_pool(out, batch=batch)


if __name__ == '__main__':
    x = torch.randn([10, 116, 110])
    x = x.mean(dim=2)
    a = torch.corrcoef(x.transpose(1, 0))


