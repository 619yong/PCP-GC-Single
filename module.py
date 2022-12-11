import torch
import numpy as np
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch
import dgl.nn.pytorch.conv
import math
import torch.nn as nn


from GNN_encoder import SAGE,EdgeUpdate,NKMLPPredictor
from decode import Block_ResNet_Module

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()

        self.feed1 = torch.nn.Linear(d_model, d_ff)
        self.bn1 = torch.nn.BatchNorm1d(num_features=d_ff)
        self.relu1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(dropout)

        self.feed2 = torch.nn.Linear(d_ff, d_model)
        self.bn2 = torch.nn.BatchNorm1d(num_features=d_model)
        self.relu2 = torch.nn.ReLU()
        self.drop2 = torch.nn.Dropout(dropout)

    def forward(self, input):
        input1 = self.drop1(self.relu1(self.bn1(self.feed1(input))))
        input2 = self.drop2(self.relu2(self.bn2(self.feed2(input1))))

        return input2


class GAU(nn.Module):
    def __init__(
        self,
        dim,
        query_key_dim = 128,
        expansion_factor = 2.,
        add_residual = True,
        dropout = 0.,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )

        self.gamma = nn.Parameter(torch.ones(2, query_key_dim))
        self.beta = nn.Parameter(torch.zeros(2, query_key_dim))
        nn.init.normal_(self.gamma, std=0.02)


        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.add_residual = add_residual

    def forward(self, x):
        seq_len = x.shape[-2]

        normed_x = self.norm(x) #(bs,seq_len,dim)
        v, gate = self.to_hidden(normed_x).chunk(2, dim = -1) #(bs,seq_len,seq_len)

        Z = self.to_qk(normed_x) #(bs,seq_len,query_key_dim)

        QK = torch.einsum('... d, h d -> ... h d', Z, self.gamma) + self.beta
        q, k = QK.unbind(dim=-2)

        sim = torch.einsum('b i d, b j d -> b i j', q, k) / seq_len

        A = F.relu(sim) ** 2
        A = self.dropout(A)

        V = torch.einsum('b i j, b j d -> b i d', A, v)
        V = V * gate

        out = self.to_out(V)
        if self.add_residual:
            out = out + x
        return out


class SAGE_Module_with_DeepGCN(torch.nn.Module):
    def __init__(self, seq_feats, edges_feats, hidden1_feats, hidden2_feats, out_feats, num, is_contact, gau,
                 need_change):
        super(SAGE_Module_with_DeepGCN, self).__init__()
        self.gau = gau
        self.need_change = need_change
        self.linear1_1 = torch.nn.Linear(seq_feats, hidden1_feats)
        self.bn1 = torch.nn.BatchNorm1d(num_features=hidden1_feats)
        self.relu1 = torch.nn.ReLU()  # 原来sage得到的激活函数是relu，现在改为leakyrelu,leakyreul 效果不好
        # self.relu1 = torch.nn.LeakyReLU()
        self.sage1 = SAGE(in_feat=hidden1_feats, hidden_feat=hidden1_feats * 2, out_feat=hidden1_feats, dropout=0.2)
        # self.linear1_2 = torch.nn.Linear(hidden1_feats,hidden1_feats)
        if self.gau == True:
            self.linear1_2 = GAU(dim=hidden1_feats, query_key_dim=int(hidden1_feats / 4), expansion_factor=2)
        else:
            self.linear1_2 = GAU(dim=hidden1_feats, query_key_dim=int(hidden1_feats * 2), expansion_factor=2)

        self.linear2_1 = torch.nn.Linear(hidden1_feats, hidden1_feats)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden1_feats)
        self.relu2 = torch.nn.ReLU()  # 上同
        # self.relu2 = torch.nn.LeakyReLU()
        self.sage2 = SAGE(in_feat=hidden1_feats, hidden_feat=hidden2_feats, out_feat=hidden2_feats, dropout=0.3)
        # self.linear2_2 = torch.nn.Linear(hidden2_feats,hidden2_feats)
        if self.gau == True:
            self.linear2_2 = GAU(dim=hidden2_feats, query_key_dim=int(hidden2_feats / 4), expansion_factor=2)

        self.edges1 = EdgeUpdate(in_feats=hidden2_feats, edge_infeat=edges_feats,
                                 hid_feats=hidden1_feats * 2,
                                 out_feats=hidden2_feats)

        self.linear3 = GAU(dim=hidden2_feats, query_key_dim=int(hidden2_feats / 4), expansion_factor=2)

        self.nk1 = NKMLPPredictor(in_features=hidden2_feats, edges_feature=hidden2_feats + edges_feats,
                                  hidden_feature=hidden2_feats * 2, out_classes=out_feats)

        self.conv2d = Block_ResNet_Module(out_feats + edges_feats + hidden2_feats, num, True)
        self.c1 = nn.Conv2d(in_channels=out_feats + edges_feats + hidden2_feats, out_channels=out_feats, kernel_size=1,
                            stride=1, bias=True)

        self.b1 = nn.BatchNorm2d(num_features=out_feats)
        self.l1 = nn.LeakyReLU()
        self.c2 = nn.Conv2d(in_channels=out_feats, out_channels=1, kernel_size=1, stride=1, bias=True)

        self.is_contact = is_contact
        if self.is_contact == True:
            self.b2 = nn.BatchNorm2d(num_features=1)
            self.l2 = nn.Sigmoid()
            # self.l2 = nn.LogSigmoid()
        else:  # 实值预测
            # self.dropout = nn.Dropout(0.1)
            self.l2 = nn.Sigmoid()
            self.b2 = nn.BatchNorm2d(num_features=1)

    def forward(self, graph, nodes, gk):
        L = nodes.shape[0]
        if self.need_change == False:
            nodes1 = self.relu1(self.bn1(self.linear1_1(nodes)))
            nodes1 = self.sage1(gk, nodes1)
            nodes1 = self.linear1_2(nodes1.unsqueeze(0)).squeeze(0)  # (L,hidden1)

            nodes2 = self.relu2(self.bn2(self.linear2_1(nodes1)))
            nodes2 = self.sage2(gk, nodes2)
            nodes2 = self.linear2_2(nodes2.unsqueeze(0)).squeeze(0)  # (L,hidden2)

            edges_feature = graph.edata['feature'].reshape(L, L, -1)

            graph.edata['feature'] = torch.cat((graph.edata['feature'], self.edges1(graph, nodes2)), dim=-1)

            nodes3 = self.linear3(nodes2.unsqueeze(0)).squeeze(0)
            score = self.nk1(graph, nodes3).reshape(L, L, -1)

        nodes4 = torch.einsum('ik,jk->ijk', nodes3, nodes3)
        score1 = torch.cat((score, edges_feature, nodes4), dim=-1)

        score1 = self.conv2d(score1)
        score1 = score1.unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
        score1 = self.l1(self.b1(self.c1(score1)).squeeze(0).permute(1, 2, 0).squeeze(2))

        # score = self.conv2d(score)
        score1 = score1.unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
        if self.is_contact:
            score1 = self.l2(self.b2(self.c2(score1)).squeeze(0).permute(1, 2, 0).squeeze(2))
        else:

            score1 = self.l2(self.b2(self.c2(score1)).squeeze(0).permute(1, 2, 0).squeeze(2))
            # score1 = self.l2(self.c2(score1).squeeze(0).permute(1,2,0).squeeze(2))
            score1 = (score1 + score1.permute(1, 0)) / 2
        return nodes4, score, score1


class SAGE_Predictor(torch.nn.Module):
    def __init__(self, amio_feature, in_feats, hidden_feats1, hidden_feats2, out_feats, device='cuda:1', num=5,
                 is_contact=True, gau=True, need_change=False, model=None, need_onehot=True, ablation=False):
        super(SAGE_Predictor, self).__init__()
        self.nodes_model = torch.nn.Linear(amio_feature + 1280, in_feats)

        # self.nodes_model = torch.nn.Linear(amio_feature,in_feats)
        self.pcm = SAGE_Module_with_DeepGCN(seq_feats=in_feats, edges_feats=20, hidden1_feats=hidden_feats1,
                                            hidden2_feats=hidden_feats2, out_feats=out_feats, num=num,
                                            is_contact=is_contact, gau=gau, need_change=need_change)
        if need_change == True:
            self.pcm = SAGE_Module_with_Recurt(seq_feats=in_feats, edges_feats=20, hidden1_feats=hidden_feats1,
                                               hidden2_feats=hidden_feats2, out_feats=out_feats, num=num,
                                               is_contact=is_contact, gau=gau, need_change=False)
        if need_onehot == False:
            self.nodes_model = torch.nn.Linear(1280, in_feats)
            self.pcm = SAGE_Module_with_DeepGCN(seq_feats=in_feats, edges_feats=20, hidden1_feats=hidden_feats1,
                                                hidden2_feats=hidden_feats2, out_feats=out_feats, num=num,
                                                is_contact=is_contact, gau=gau, need_change=need_change)
        self.device = device
        self.ablation = ablation
        self.need_change = need_change
        self.model = model
        self.need_onehot = need_onehot

    def forward(self, seq, feat, attention_map, score):
        seq_onehot = torch.nn.functional.one_hot(seq, num_classes=20)
        # nodes = torch.cat([seq_onehot, feat], dim=-1).to(self.device)

        if self.need_onehot == False:
            nodes = torch.nn.functional.leaky_relu(self.nodes_model(feat)).to(self.device)
        # nodes = seq_onehot.to(torch.float32)
        else:
            nodes = torch.cat([seq_onehot, feat], dim=-1).to(self.device)
            nodes = self.nodes_model(nodes).to(self.device)
        L = nodes.shape[0]
        idx = torch.ones(size=(L, L))
        src, dst = torch.where(idx == 1)
        graph = dgl.graph((src, dst), num_nodes=L).to(self.device)
        attention_map = attention_map.to(self.device)

        graph.edata['feature'] = attention_map[src, dst]

        values, index = torch.topk(score.unsqueeze(0), dim=-1, k=min(L, 40))
        if self.ablation == True: values, index = torch.topk(score.unsqueeze(0), dim=-1, k=L)

        topk_matric = torch.zeros(size=(1, L, L)).to(self.device)
        topk_matric.scatter_(2, index, 1.0)
        topk_matric = topk_matric.squeeze(0)
        i, j = torch.where(topk_matric)
        gk = dgl.graph((i, j), num_nodes=L)
        gk = gk.to(self.device)

        new_nodes, edges, score = self.pcm(graph, nodes, gk)
        return new_nodes, edges, score
