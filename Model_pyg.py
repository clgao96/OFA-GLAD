import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from utils_pyg import custom_cosine_loss, loss_func_feat, loss_func_adj
from torch_geometric.nn import GCNConv, GINConv, global_add_pool , global_mean_pool, BatchNorm, GraphNorm
from torch_geometric.utils import scatter

class Encoder_GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers, drop_rate, is_bn=False, is_norm=False):
        super(Encoder_GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.normalize = is_norm
        self.drop = nn.Dropout(p=drop_rate)
        self.act = nn.ReLU()
        self.is_bn = is_bn
        self.bn = BatchNorm(embedding_dim)

        for i in range(self.num_layers):
            if i == 0:
                conv = GCNConv(input_dim, embedding_dim)
            elif 1 <= i < self.num_layers - 1:
                conv = GCNConv(embedding_dim, embedding_dim)
            else:
                conv = GCNConv(embedding_dim, embedding_dim)
            self.convs.append(conv)

    def forward(self, x, edge_index, batch):
        h = self.act(self.convs[0](x, edge_index))
        h = self.drop(h)
        if self.normalize:
            h = F.normalize(h, p=2, dim=1)
        if self.is_bn:
            h = self.bn(h)
        for i in range(1, self.num_layers):
            h = self.act(self.convs[i](h, edge_index))
            if 1<= i < self.num_layers - 1:
                h = self.drop(h)
            if self.normalize:
                h = F.normalize(h, p=2, dim=1)
            if self.is_bn:
                h = self.bn(h)
        xpool = global_add_pool(h, batch)

        return h , xpool


class attr_Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, dropout):
        super(attr_Decoder, self).__init__()

        self.lin1 = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.act(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return x

class stru_Decoder(nn.Module):
    def __init__(self):
        super(stru_Decoder, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adj):
        b, n, n = adj.shape
        x = x.view(b, n, -1)
        adj_recon = torch.bmm(x, x.transpose(1, 2))
        adj_recon = nn.ReLU()(adj_recon)
        return adj_recon

class GNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers, drop_1  #, drop_2, prompt_dim
                 , bn=True, norm=True):
        super(GNNet, self).__init__()
        self.encoder = Encoder_GCN(input_dim, hidden_dim, embedding_dim, num_layers, drop_1, is_bn=bn, is_norm=norm)
        self.feat_dec = attr_Decoder(input_dim, hidden_dim, embedding_dim, drop_1)
        self.adj_dec = stru_Decoder()

    def forward(self, g):
        node_embed, graph_embed = self.encoder(g.x, g.edge_index, g.batch)
        recon_node = self.feat_dec(node_embed, g.edge_index)
        recon_adj = self.adj_dec(node_embed, g.adj)

        return node_embed, recon_node, recon_adj


class CrossAttn(nn.Module):
    def __init__(self, prompt_dim, input_dim, embedding_dim, drop_rate, lammda , alpha, beta): # , alpha, beta
        super(CrossAttn, self).__init__()
        self.input_dim = input_dim
        self.prompt_dim = prompt_dim
        self.lammda = lammda
        self.alpha = alpha
        self.beta = beta

        self.mlp_p = nn.Sequential(
            nn.Linear(prompt_dim, embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.mlp_c = nn.Sequential(
            nn.Linear(self.input_dim * 8, embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh()
        )

        self.mlp_q = nn.Sequential(
        nn.Linear(self.input_dim * 8, embedding_dim),
        nn.LeakyReLU(0.2),
        )
        self.lin_1 = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim * 2),
            nn.LeakyReLU(0.2),
        )
        self.lin_2 = nn.Sequential(
            nn.Linear(1, self.input_dim * 2),
            nn.LeakyReLU(0.2),
        )


    def get_train_loss_error_distribution(self, recon_feat, recon_adj, g, num_prompt):
        h0 = g.x
        adj = g.adj
        b, n, n = adj.shape

        context_batch_indices = torch.randperm(recon_adj.size(0))[:num_prompt].to(recon_adj.device)
        query_batch_indices = torch.tensor([i for i in range(recon_adj.size(0)) if i not in context_batch_indices.tolist()]).to(recon_adj.device)

        error_feat = torch.abs(h0 - recon_feat).view(b, n, self.input_dim)
        error_adj = torch.abs(adj - recon_adj).sum(dim=-1).unsqueeze(-1)

        error_feat = self.lin_1(error_feat)
        error_adj = self.lin_2(error_adj)

        avg_feat = error_feat.mean(dim=1)
        std_feat = error_feat.std(dim=1)
        avg_adj = error_adj.mean(dim=1)
        std_adj = error_adj.std(dim=1)
        error_distribution = torch.cat([avg_feat, std_feat, avg_adj, std_adj], dim=-1)
        query_error = error_distribution[query_batch_indices]

        context_error = error_distribution[context_batch_indices]
        query_error = self.mlp_q(query_error)
        z = torch.randn(self.prompt_dim).to(query_error.device)
        prop = self.mlp_p(z)
        p_prompt = self.mlp_c(context_error).mean(dim=0)
        prop_tilde = prop + self.lammda * p_prompt

        loss_graph = torch.norm(query_error - prop_tilde, p=2, dim=1)
        loss = loss_graph.mean()

        return loss

    def get_test_score_error_distribution(self, recon_feat, recon_adj, g, num_prompt, return_components=False):
        h0 = g.x
        adj = g.adj
        b, n, n = adj.shape

        query_batch_indices = torch.arange(recon_adj.size(0) - num_prompt).to(recon_adj.device)
        context_batch_indices = torch.arange(recon_adj.size(0) - num_prompt, recon_adj.size(0)).to(recon_adj.device)

        error_feat = torch.abs(h0 - recon_feat).view(b, n, self.input_dim)
        error_adj = torch.abs(adj - recon_adj).sum(dim=-1).unsqueeze(-1)

        error_feat = self.lin_1(error_feat)
        error_adj = self.lin_2(error_adj)


        avg_feat = error_feat.mean(dim=1)
        std_feat = error_feat.std(dim=1)
        avg_adj = error_adj.mean(dim=1)
        std_adj = error_adj.std(dim=1)
        error_distribution = torch.cat([avg_feat, std_feat, avg_adj, std_adj], dim=-1)

        query_error = error_distribution[query_batch_indices]
        context_error = error_distribution[context_batch_indices]
        query_error = self.mlp_q(query_error)

        z = torch.randn(self.prompt_dim).to(query_error.device)
        prop = self.mlp_p(z)
        p_prompt = self.mlp_c(context_error).mean(dim=0)
        prop_tilde = prop + self.lammda * p_prompt

        query_score_graph = torch.norm(query_error - prop_tilde, p=2, dim=1)

        _, loss_recon_feat = loss_func_feat(h0, recon_feat, g.batch)
        _, loss_recon_adj = loss_func_adj(adj, recon_adj)
        query_score_feat = loss_recon_feat[query_batch_indices]
        query_score_adj = loss_recon_adj[query_batch_indices]

        query_score = query_score_graph + query_score_feat + query_score_adj

        return query_score
