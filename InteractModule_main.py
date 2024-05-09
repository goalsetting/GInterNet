import math

from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

import torch

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, features, dropout):
        super(MLP, self).__init__()
        self.features = features
        self.linear1 = torch.nn.Linear(nfeat, nhid)
        self.linear2 = torch.nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self):
        x = self.features
        x = F.relu(self.linear1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)

class HGraphAttentionLayer(nn.Module):
    def __init__(self, out_features, dropout, alpha, threshold=0.2, concat=True):
        super(HGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.threshold = threshold

        self.a = nn.Parameter(torch.empty(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, h2,adjs):
        Wh = h  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh,h2)
        zero_vec = -9e15 * torch.ones_like(e)
        attentions=[]
        for i in range(len(adjs)):
            attention = torch.where(adjs[i] > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            attentions.append(attention)
        return attentions

    def _prepare_attentional_mechanism_input(self, Wh,Whother):
        Wh1 = torch.matmul(Wh, self.a[:Wh.shape[1], :])
        Wh2 = torch.matmul(Whother, self.a[Wh.shape[1]:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphInter_layer(Module):
    def __init__(self, raw_infeatures, out_features, threshold=0.0, label=[]):
        super(GraphInter_layer, self).__init__()
        self.in_features = raw_infeatures
        self.out_features = out_features
        self.modellayers = nn.Sequential()
        self.all_features = sum(raw_infeatures)
        self.label = label
        self.W = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.all_features, self.out_features)) for _ in range(len(raw_infeatures))])
        self.threshold = threshold
        self.reset_parameters()
        for i in range(len(raw_infeatures)):
            self.modellayers.add_module(f'att{i}',HGraphAttentionLayer(self.all_features,
                                                dropout = 0.5,
                                                 alpha = 0.2,
                                                threshold = threshold,
                                                 concat = False))


    def reset_parameters(self):
        for i in range(len(self.W)):
            stdv = 1. / math.sqrt(self.W[i].size(1))
            self.W[i].data.uniform_(-stdv, stdv)

    def loss_GInterNet(self, Ws):
        same_class_mask = (self.label.unsqueeze(1) == self.label.unsqueeze(0))
        Ws = torch.stack(Ws)
        potential_loss = torch.sum(Ws * same_class_mask) / torch.sum(Ws)
        return potential_loss

    def forward(self, features, adjs):
        x = []
        potential_loss = []
        learned_adjs = []
        for i in range(len(features)):
            a = features[i]
            b = torch.cat([features[j] for j in range(len(features)) if j != i],dim=1)
            tmpadjs = [element for index, element in enumerate(adjs) if index != i]
            interadj = self.modellayers[i](a, b, tmpadjs)
            potential_loss.append(self.loss_GInterNet(interadj))
            learned_adjs.extend(interadj)
            if len(tmpadjs)==1:
                tmpx = torch.cat((a,torch.spmm(interadj[0], b)), dim=1)
            else:
                tmpx = torch.cat(
                    (a, torch.stack([torch.spmm(interadj[jj], b) for jj in range(len(tmpadjs)) if jj != i]).mean(0)),
                    dim=1)
            tmpx = torch.mm(tmpx,self.W[i])
            x.append(tmpx)

        return x, potential_loss, learned_adjs

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GInterNet(Module):
    def __init__(self, hid_dim, out_features, features, adjs, dropout=0.5, label=[], k=2, unlabeled_index = []):
        super(GInterNet, self).__init__()
        self.features = features
        self.adjs = adjs
        self.k = k
        self.in_features = [self.features[i].shape[1] for i in range(len(self.features))]        # [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        self.gis = [GraphInter_layer(self.in_features, hid_dim, 0, label)]
        for i in range(k-1):
            self.gis.append(GraphInter_layer([hid_dim for _ in range(len(self.features))], hid_dim, 0, label))
        self.gis = nn.ModuleList(self.gis)
        self.gc = torch.nn.Linear(hid_dim*len(self.features), hid_dim, bias=False)
        self.gc1 = torch.nn.Linear(hid_dim, out_features, bias=False)
        self.dropout = dropout
        self.unlabeled_index = unlabeled_index


    def loss_unlabel(self, adj, x, unlabeled_index):
        adj_tmp = torch.stack(adj)
        x = F.normalize(x)
        cos = torch.mm(x, x.t())
        a = adj_tmp[:, unlabeled_index,unlabeled_index]
        return torch.mul(a, cos[unlabeled_index,unlabeled_index]).sum()/a.sum()

    def forward(self):
        x = self.features
        true_loss = []
        all_adjs = []
        for i in range(self.k):
            x, loss_tmp, learned_adjs = self.gis[i](x, self.adjs)
            all_adjs.extend(learned_adjs)
            true_loss.append(torch.stack(loss_tmp).mean())
            for j in range(len(x)):
                x[j] = F.relu(x[j])
                x[j] = F.dropout(x[j], self.dropout, training=self.training)
        x = torch.cat(x, dim=1)
        x = F.relu(self.gc(x))
        x = self.gc1(x)
        un_loss = self.loss_unlabel(all_adjs, F.log_softmax(x, dim=1), self.unlabeled_index)
        return F.log_softmax(x, dim=1), true_loss, un_loss