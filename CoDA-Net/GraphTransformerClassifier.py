import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, TransformerConv, GCNConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch

from BrainNodeAttention import NodeAttention
from GCT import GraphCTransformerLayer, GraphTransformerLayer


def create_batch(x, adj):
    data_list = []
    for i in range(x.size(0)):
        edge_index, _ = dense_to_sparse(adj[i])
        data = Data(x=x[i], edge_index=edge_index)
        data_list.append(data)
    return Batch.from_data_list(data_list)


class GraphTransformerExtractor(nn.Module):
    def __init__(self, in_dim, hidden_dim=None):
        super(GraphTransformerExtractor, self).__init__()
        if hidden_dim is None:
            hidden_dim = [32, 64, 128, 256, 512]
        self.input_proj = nn.Linear(in_dim, hidden_dim[0])

        self.layer1 = GraphCTransformerLayer(hidden_dim[0], hidden_dim[1], heads=8)
        self.layer2 = GraphCTransformerLayer(hidden_dim[1], hidden_dim[2], heads=4)
        self.layer3 = GraphCTransformerLayer(hidden_dim[2], hidden_dim[3], heads=2)
        self.layer4 = GraphCTransformerLayer(hidden_dim[3], hidden_dim[4], heads=1)
        self.ba = NodeAttention(hidden_dim[4])
        self.dropout = nn.Dropout(0.5)
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
                # no bias to init

            elif isinstance(m, GCNConv):
                nn.init.xavier_uniform_(m.lin.weight)
                # no bias property on GCNConv.lin

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)

        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)
        x = self.layer4(x, edge_index)

        attention, x_atten = self.ba(x, edge_index, batch)
        x = torch.cat([x, x_atten], dim=1)
        return global_mean_pool(x, batch)


class GraphTransformerClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, num_classes = 2):
        super(GraphTransformerClassifier, self).__init__()
        if hidden_dim is None:
            hidden_dim = [32, 64, 128, 256, 512]
        self.input_proj = nn.Linear(in_dim, hidden_dim[0])

        self.layer1 = GraphCTransformerLayer(hidden_dim[0], hidden_dim[1], heads=8)
        self.layer2 = GraphCTransformerLayer(hidden_dim[1], hidden_dim[2], heads=4)
        self.layer3 = GraphCTransformerLayer(hidden_dim[2], hidden_dim[3], heads=2)
        self.layer4 = GraphCTransformerLayer(hidden_dim[3], hidden_dim[4], heads=1)
        self.ba = NodeAttention(hidden_dim[4])

        # self.fc = nn.Linear(hidden_dim[-1], num_classes)
        self.fc1 = nn.Linear(hidden_dim[-1], num_classes)
        self.fc2 = nn.Linear(116, num_classes)
        self.fc_final = nn.Linear(hidden_dim[-1] + 116, num_classes)
        # self.fc_final = nn.Linear(hidden_dim[-1], num_classes)
        self.dropout = nn.Dropout(0.5)

    def classifier(self, x):
        logits = self.fc_final(x)
        return logits

    def feature_forward(self, x, edge_index, batch):
        x = self.input_proj(x)

        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)
        x = self.layer4(x, edge_index)

        attention, x_atten = self.ba(x, edge_index, batch)
        x = torch.cat([x, x_atten], dim=1)
        return global_mean_pool(x, batch), attention

    def forward(self, x, adj):
        batch = create_batch(x, adj)
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        feature, attention = self.feature_forward(x, edge_index, batch)
        logit = self.classifier(feature)
        return attention, logit


if __name__ == '__main__':
    x = torch.randn([10, 116, 110])
    c = torch.zeros(x.shape[0]).long()
    c = F.one_hot(c, 2)
    t = torch.randint(0, 100, (10,)).long()
    # model = TransformerDenoiser(110, 116, 112, head=4)
    # print(model(x, c, t).shape)

