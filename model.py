import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.attn = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.attn.data)

    def forward(self, x, adj):
        Wh = self.W(x)
        N = Wh.size(0)
        a_input = torch.cat([
            Wh.repeat(1, N).view(N * N, -1),
            Wh.repeat(N, 1)
        ], dim=1).view(N, N, 2 * Wh.size(1))
        e = F.leaky_relu(torch.matmul(a_input, self.attn).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)


class GATPredictor(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.gat1 = GATLayer(in_features, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_features)

    def forward(self, x, adj):
        h = self.gat1(x, adj)
        return self.fc(h)
