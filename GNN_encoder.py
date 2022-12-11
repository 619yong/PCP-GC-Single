import torch
import numpy as np
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch
import dgl.nn.pytorch.conv
import math
import torch.nn as nn

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



class EdgeUpdate(nn.Module):
    def __init__(self, in_feats, edge_infeat, hid_feats,out_feats):
        super().__init__()
        self.node1_model = FeedForward(d_model=in_feats,d_ff=in_feats*2,dropout=0.1)
        self.node2_model = FeedForward(d_model=in_feats,d_ff=in_feats*2,dropout=0.1)

        self.W1 = nn.Linear(in_feats * 2 + edge_infeat, hid_feats*2)
        self.bn1 = torch.nn.BatchNorm1d(num_features=hid_feats*2)
        self.relu1 = torch.nn.LeakyReLU()

        self.W2 = nn.Linear(hid_feats*2, hid_feats)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hid_feats)
        self.relu2 = torch.nn.LeakyReLU()

        self.W3 = nn.Linear(hid_feats,out_feats)
        self.bn3 = torch.nn.BatchNorm1d(num_features=out_feats)
        self.relu3 = torch.nn.LeakyReLU()
        # self.W3 = nn.Linear(10, 4)

    def apply_edges(self, edges):
        h_u = self.node1_model(edges.src['h'])
        h_v = self.node2_model(edges.dst['h'])
        # print("h_v",h_v.size())
        edgesfeature = torch.cat([h_u, h_v, edges.data['feature']], 1)
        # print(edgesfeature.size())

        e_feats = self.relu1(self.bn1(self.W1(edgesfeature)))
        e_feat = self.relu2(self.bn2(self.W2(e_feats)))
        e = self.relu3(self.bn3(self.W3(e_feat)))
        # e_feats = F.relu(self.W3(e_feats))
        return {'feature': e}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['feature']

class NKMLPPredictor(nn.Module):
    def __init__(self, in_features,edges_feature,hidden_feature, out_classes):
        super().__init__()
        self.n1 = FeedForward(d_model=in_features,d_ff=in_features*2,dropout=0.1)
        self.n2 = FeedForward(d_model=in_features,d_ff=in_features*2,dropout=0.1)

        self.W1 = nn.Linear(in_features * 2 + edges_feature, hidden_feature*2, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_feature*2)
        self.relu1 = torch.nn.LeakyReLU()

        self.W2 = nn.Linear(hidden_feature*2, hidden_feature)
        self.bn2 = nn.BatchNorm1d(hidden_feature)
        self.relu2 = torch.nn.LeakyReLU()

        self.W3 = nn.Linear(hidden_feature, out_classes)
        self.bn3 = nn.BatchNorm1d(num_features=out_classes)
        self.relu3 = nn.LeakyReLU()

    def apply_edges(self, edges):
        h_u = self.n1(edges.src['h'])
        h_v = self.n2(edges.dst['h'])
        # score = self.W(torch.cat([h_u, h_v, edges.data['feature']], 1))
        # score = self.W2(score)
        edgesfeature = torch.cat([h_u, h_v, edges.data['feature']], 1)
        edges_feats = self.relu1(self.bn1(self.W1(edgesfeature)))
        edges_feat = self.relu2(self.bn2(self.W2(edges_feats)))
        score = self.relu3(self.bn3(self.W3(edges_feat)))

        return {'score': score}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)

            return g.edata['score']


class SAGE(torch.nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, dropout):
        super(SAGE, self).__init__()
        # print(in_feat,hidden_feat,dropout)

        self.feed = FeedForward(d_model=in_feat, d_ff=hidden_feat, dropout=dropout)

        self.sage1 = dgl.nn.pytorch.conv.SAGEConv(in_feats=in_feat, out_feats=hidden_feat, aggregator_type='mean')
        # self.sage1 = dgl.nn.pytorch.conv.GATConv(in_feats=in_feat,out_feats=hidden_feat,num_heads=1,feat_drop=0.1,attn_drop=0.1)
        self.bn1d1 = torch.nn.BatchNorm1d(num_features=hidden_feat)
        self.activation1 = torch.nn.LeakyReLU()
        # self.activation1 = torch.nn.ReLU()
        self.sage2 = dgl.nn.pytorch.conv.SAGEConv(in_feats=hidden_feat, out_feats=out_feat, aggregator_type='mean')
        # self.sage2 = dgl.nn.pytorch.conv.GATConv(in_feats=hidden_feat,out_feats=out_feat,num_heads=1,feat_drop=0.1,attn_drop=0.1)
        self.bn1d2 = torch.nn.BatchNorm1d(num_features=out_feat)
        self.activation2 = torch.nn.LeakyReLU()
        # self.activation2 = torch.nn.ReLU()

    def forward(self, graph, nodes):
        L = nodes.shape[0]
        node = self.feed(nodes)
        # print('nodes:',nodes)
        nodes_gcn1 = self.sage1(graph, node).reshape(L, -1)
        # print('nodes_gcn1:',nodes_gcn1.shape)
        nodes_ret1 = self.activation1(self.bn1d1(nodes_gcn1))
        # print('nodes_ret1:',nodes_ret1)
        nodes_gcn2 = self.sage2(graph, nodes_ret1).reshape(L, -1)
        # print('nodes_gcn2:',nodes_gcn2.shape)
        nodes_ret2 = self.activation2(self.bn1d2(nodes_gcn2))

        return nodes_ret2








