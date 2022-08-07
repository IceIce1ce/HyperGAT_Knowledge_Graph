import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable

CUDA = torch.cuda.is_available()

class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()
        self.hidden_size = input_dim
        self.fil_size = 1
        self.dropout_rate =  drop_prob
        self.out_channels = out_channels
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.fc_r = nn.Linear(self.hidden_size, self.out_channels*3*self.fil_size)
        l = input_dim*2
        self.fc_1 = nn.Linear((self.hidden_size-self.fil_size + 1)*self.out_channels, l)
        self.fc_r2 = nn.Linear(self.hidden_size, l)
        self.in_drop = nn.Dropout(0.2)
        self.w_drop = nn.Dropout(0.4)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.ReLU()
        self.elu = nn.ELU()
        nn.init.xavier_normal_(self.fc_r.weight.data, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_normal_(self.fc_r2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc_1.weight.data, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_uniform_(self.fc_r.weight, gain=1.414)
        #nn.init.xavier_uniform_(self.fc_r2.weight, gain=1.414)
        #nn.init.xavier_uniform_(self.fc_1.weight, gain=1.414)

    def forward(self, h, r, t):
        bs = h.size(0)
        I = r
        h = h.unsqueeze(1)
        t = t.unsqueeze(1)
        r_ = r.unsqueeze(1)
        E = torch.cat([h, r_, t], dim=1)
        E = E.transpose(1, 2)
        E = E.unsqueeze(0)
        R = self.fc_r(I)
        R = self.tanh(R)
        R = R.view(bs*self.out_channels, 1, self.fil_size, 3)
        X = F.conv2d(E, R, groups=bs)
        X = self.relu(X)
        X = X.view(bs, self.out_channels, -1, 1)
        X = self.bn(X)
        X = X.view(bs, -1)
        X = self.dropout(X)
        X = self.fc_1(X)
        X = self.elu(X)
        X = X.unsqueeze(-1)
        W = self.fc_r2(I)
        W = self.elu(W)
        W = W.unsqueeze(1)
        output = torch.matmul(W, X).view(-1)
        return output

class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        # assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim

        self.a = nn.Parameter(torch.zeros(
            size=(out_features, 2 * in_features + nrela_dim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, edge, edge_embed, edge_list_nhop, edge_embed_nhop):
        N = input.size()[0]

        # Self-attention on the nodes - Shared attention mechanism
        edge = torch.cat((edge[:, :], edge_list_nhop[:, :]), dim=1)
        edge_embed = torch.cat(
            (edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)

        edge_h = torch.cat(
            (input[edge[0, :], :], input[edge[1, :], :], edge_embed[:, :]), dim=1).t()
        # edge_h: (2*in_dim + nrela_dim) x E

        edge_m = self.a.mm(edge_h)
        # edge_m: D * E

        # to be checked later
        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())
        edge_e = torch.exp(powers).unsqueeze(1)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm_final(
            edge, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D

        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.shape[0], self.out_features)

        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out

        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
