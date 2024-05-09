import os

import numpy as np
import scipy.sparse as sp
import torch
import sys
import networkx as nx
import pickle as pkl
import math

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))
#这个函数为辅助韩书记和，比如读边，读数据

def Build_ADJ(data):
    adj = sp.coo_matrix((np.ones(data.num_edges), (data.edge_index[0], data.edge_index[1])),
                        shape=(data.num_nodes, data.num_nodes),
                        dtype=np.float32)
    adj_ori = torch.Tensor(adj.toarray())
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    adj_label = adj_ori + torch.eye(adj_ori.shape[0])

    pos_weight = (adj_ori.shape[0] ** 2 - adj_ori.sum()) / adj_ori.sum()
    norm = adj_ori.shape[0] ** 2 / (2 * (adj_ori.shape[0] ** 2 - adj_ori.sum()))
    return adj, adj_ori, adj_label, pos_weight, norm


def Reconstruct_Ratio(pred, adj_ori):
    adj_pred = pred.reshape(-1)
    adj_pred = (sigmoid(adj_pred) > 0.5).float()
    adj_true = (adj_ori + torch.eye(adj_ori.shape[0])).reshape(-1)
    adj_acc = float(adj_pred.eq(adj_true).sum().item()) / adj_pred.shape[0]
    return adj_acc

def get_loaclitems(adjtensor):
    localadjtensors = []
    # 获得所有的局部组合
    for k in range(len(adjtensor) - 1):
        for kk in range(k + 1, len(adjtensor)):
            localadjtensor = []
            localadjtensor.append(adjtensor[k])
            localadjtensor.append(adjtensor[kk])
            localadjtensors.append(localadjtensor)
    return localadjtensors

def load_adjs(path="../data/cora/", dataset="cora"):


    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    adj2 = sp.coo_matrix(adj * adj)
    adj3 = sp.coo_matrix(adj2 * adj)
    adj4 = sp.coo_matrix(adj3 * adj)
    adj5 = sp.coo_matrix(adj4 * adj)
    adj6 = sp.coo_matrix(adj5 * adj)





    sparse_mx = adj.tocoo().astype(np.float32)
    sparse_mx = adj2.tocoo().astype(np.float32)
    sparse_mx = adj3.tocoo().astype(np.float32)
    sparse_mx = adj4.tocoo().astype(np.float32)
    sparse_mx = adj5.tocoo().astype(np.float32)
    sparse_mx = adj6.tocoo().astype(np.float32)

    adj = torch.FloatTensor(np.array(adj.todense()))
    adj2 = torch.FloatTensor(np.array(adj2.todense()))
    adj3 = torch.FloatTensor(np.array(adj3.todense()))
    adj4 = torch.FloatTensor(np.array(adj4.todense()))
    adj5 = torch.FloatTensor(np.array(adj5.todense()))
    adj6 = torch.FloatTensor(np.array(adj6.todense()))

    return adj, adj2, adj3,adj4,adj5,adj6

def parse_index_file(filename):
    index = []

    for line in open(filename):
        index.append(int(line.strip()))

    return index
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_adjs(path="../data/cora/", dataset="cora",r=0):
    adj = None
    if dataset=="texas" or dataset=="cornell" or dataset=="coraml" or dataset=="chameleon" or dataset=="squirrel" or dataset=="wisconsin":
        graph_adjacency_list_file_path = path + dataset + "/out1_graph_edges.txt"
        graph_node_features_and_labels_file_path = path + dataset + "/out1_node_feature_label.txt"

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        adj  = adj  + adj .T.multiply(adj.T > adj ) - adj .multiply(adj .T > adj )
        adj = normalize(adj  + sp.eye(adj.shape[0]))
    elif dataset != "cora" and dataset != "citeseer" and dataset != "pubmed":
        names = ['edge']
        objects = []
        features = np.genfromtxt(path+"/{}.{}".format(dataset, "feature"),
                                            dtype=np.dtype(str))
        labels = np.genfromtxt(path+"/{}.{}".format(dataset, "label"),
                                            dtype=np.dtype(str))

        featuregraph_path = np.genfromtxt(path+"/{}.{}".format(dataset, names[0]),
                                        dtype=np.int32)
        feature_edges = featuregraph_path
        fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
        # fedges经验证是等同于feature_edges
        fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(len(labels), len(labels)),
                             dtype=np.float32)
        # 创建稀疏矩阵
        fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
        adj = normalize(fadj + sp.eye(fadj.shape[0]))
        # nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)


    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []

        for i in range(len(names)):
            with open(path+"ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        # graph=inject_adjnosie(graph, 0.8)
        # if r != 0:
        #     adj = inject_adjnosie(graph, r)
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = sp.csr_matrix(adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        # adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # adj2 = sp.csc_matrix(adj * adj)
    # adj3 = sp.csc_matrix(adj2* adj)
    # adj4 = sp.csc_matrix(adj3* adj)
    # adj5 = sp.csc_matrix(adj4 * adj)
    # adj6 = sp.csc_matrix(adj5 * adj)





    # sparse_mx = adj.tocoo().astype(np.float32)
    # sparse_mx = adj2.tocoo().astype(np.float32)
    # sparse_mx = adj3.tocoo().astype(np.float32)
    # sparse_mx = adj4.tocoo().astype(np.float32)
    # sparse_mx = adj5.tocoo().astype(np.float32)
    # sparse_mx = adj6.tocoo().astype(np.float32)

    # adj = torch.FloatTensor(np.array(adj.todense()))
    # adj2 = adj2.todense()
    # adj3 = adj3.todense()
    # adj4 = adj4.todense()
    # adj5 = adj5.todense()
    # adj6 = adj6.todense()

    return adj.tocsr()


# def load_data(path="../data/cora/", dataset="cora", labelRatio = "1"):
#     """
#     ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
#     ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
#     ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
#     ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
#     ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)
#
#     ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
#     ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)
#
#     ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
#     """
#     # print("\n[STEP 1]: Upload {} dataset.".format(dataset))
#
#     names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
#     objects = []
#
#     for i in range(len(names)):
#         with open("../data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))
#             else:
#                 objects.append(pkl.load(f))
#
#     x, y, tx, ty, allx, ally, graph = tuple(objects)
#
#     test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset))
#     test_idx_range = np.sort(test_idx_reorder)
#     empetySet = []
#     if dataset == 'citeseer':
#         #Citeseer dataset contains some isolated nodes in the graph
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
#         # 找到离群点
#         # test_idx_range_reorder = list(test_idx_range_full)
#         # empetySet = set(test_idx_range_reorder).difference(set(test_idx_range))
#
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range-min(test_idx_range), :] = tx
#         tx = tx_extended
#
#         ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
#         ty_extended[test_idx_range-min(test_idx_range), :] = ty
#         ty = ty_extended
#
#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#
#     features = normalize(features)
#
#     features = torch.FloatTensor(np.array(features.todense()))
#
#     labels = np.vstack((ally, ty))
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]
#
#     no_class = set(np.where(labels)[1])
#
#     def missing_elements(L):
#         start, end = L[0], L[-1]
#         return sorted(set(range(start, end+1)).difference(L))
#
#
#     if dataset == 'citeseer':
#         save_label = np.where(labels)[1]
#         L = np.sort(test_idx_reorder)
#         missing = missing_elements(L)
#
#         for element in missing:
#             save_label = np.insert(save_label, element, 0)
#
#         labels = torch.LongTensor(save_label)
#     else:
#         labels = torch.LongTensor(np.where(labels)[1])
#
#
#
#     idx = np.arange(len(labels))
#     # 去掉离群点15个
#     idx = list(set(idx).difference(empetySet))
#
#     labels[idx] = labels[idx]
#
#
#
#     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     np.random.shuffle(idx)
#
#     # print(idx)
#
#     train_size = math.ceil(len(labels) * int(labelRatio) / 100 / len(no_class))
#     #print(math.ceil(len(labels) * labelRatio / 100 / len(no_class)))
#
#     train_size = [train_size for i in range(len(no_class))]
#
#     idx_train = []
#     count = [0 for i in range(len(no_class))]
#     label_each_class = train_size
#     next = 0
#
#
#     for i in idx:
#         if count == label_each_class:
#             break
#         next += 1
#         for j in range(len(no_class)):
#
#             if j== labels[i] and count[j] < label_each_class[j]:
#                 idx_train.append(i)
#                 count[j] += 1
#
#     idx_val = idx[next:next + 500]
#     test_size = len(test_idx_range.tolist())
#     idx_test = idx[-test_size:] if test_size else idx[next:]
#
#
#     idx_unlabel = list(set(idx).difference(set(idx_train)))
#
#     # print(len(idx_unlabel))
#
#     # idx_train = range(len(y))
#     # idx_val = range(len(y), len(y)+500)
#     # idx_test = test_idx_range.tolist()
#
#
#
#     # idx_train, idx_val, idx_test, idx_unlabel = list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test,idx_unlabel]))
#     return features, labels, idx_train, idx_val, idx_test,idx_unlabel

def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(100)
    idx_val = range(1000, 1300)
    idx_test = range(1300, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)

    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    idx_train_label = idx_train[0:round(idx_train.size(0)*0.2)]
    idx_train_unlabel = idx_train[round(idx_train.size(0) * 0.2):]
    return adj, features, labels, idx_train_label, idx_train_unlabel, idx_val, idx_test

def load_data_geometric(features,labels,lr = 1):
    no_class = set(labels.numpy())

    idx = np.arange(len(labels))
    # 去掉离群点15个
    empetySet = []
    idx = list(set(idx).difference(empetySet))

    # labels = torch.FloatTensor(labels)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    np.random.shuffle(idx)

    # print(idx)

    train_size = math.ceil(lr)
    # print(math.ceil(len(labels) * labelRatio / 100 / len(no_class)))

    train_size = [train_size for i in range(len(no_class))]

    idx_train = []
    count = [0 for j in range(len(no_class))]
    label_each_class = train_size
    next = 0

    for i in idx:
        if count == label_each_class:
            break
        next += 1
        for j in range(len(no_class)):

            if j == labels[i] and count[j] < label_each_class[j]:
                idx_train.append(i)
                count[j] += 1

    idx_val = idx[next:next + int(len(labels) * 0.1)]
    test_size = 1000
    idx_test = idx[-test_size:] if test_size else idx[next:]

    idx_unlabel = list(set(idx).difference(set(idx_train)))

    features = sp.csr_matrix(features).tolil()

    # To_Features(np.array(features.todense()), "chameleon")
    features = normalize(features.astype(dtype=float))
    features = torch.FloatTensor(np.array(features.todense()))
    return features, labels, idx_train, idx_val, idx_test, idx_unlabel

def load_data3(path="../data/cora/", dataset="cora", lr = "1"):
    """
    ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
    ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
    ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
    ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)

    ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)

    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
    """
    # print("\n[STEP 1]: Upload {} dataset.".format(dataset))
    if dataset=="texas" or dataset=="cornell" or dataset=="coraml" or dataset=="chameleon" or dataset=="squirrel" or dataset=="wisconsin":
        graph_adjacency_list_file_path = path + dataset +"/out1_graph_edges.txt"
        graph_node_features_and_labels_file_path = path + dataset + "/out1_node_feature_label.txt"

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
        features = sp.csr_matrix((features)).tolil()

        # To_Features(np.array(features.todense()), "chameleon")
        features = normalize(features.astype(dtype=float))
        features = torch.FloatTensor(np.array(features.todense()))

        no_class = set(labels)
        idx = np.arange(len(labels))
        # 去掉离群点15个
        empetySet=[]
        idx = list(set(idx).difference(empetySet))

        labels = torch.FloatTensor(labels)




        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        np.random.shuffle(idx)

        # print(idx)

        train_size = math.ceil(lr)
        #print(math.ceil(len(labels) * labelRatio / 100 / len(no_class)))

        train_size = [train_size for i in range(len(no_class))]

        idx_train = []
        count = [0 for i in range(len(no_class))]
        label_each_class = train_size
        next = 0


        for i in idx:
            if count == label_each_class:
                break
            next += 1
            for j in range(len(no_class)):

                if j== labels[i] and count[j] < label_each_class[j]:
                    idx_train.append(i)
                    count[j] += 1

        idx_val = idx[next:next + int(len(labels)*0.1)]
        test_size = 1000
            # int((len(labels)-len(idx_val)-len(idx_train))*0.6)
        idx_test = idx[-test_size:] if test_size else idx[next:]


        idx_unlabel = list(set(idx).difference(set(idx_train)))
    elif dataset!="cora" and dataset!="citeseer" and dataset!="pubmed":
        features = np.genfromtxt("{}{}.feature".format(path, dataset),
                                 dtype=np.dtype(str))
        features = np.array(features, dtype=float)
        features = sp.csr_matrix((features)).tolil()

        # features = torch.FloatTensor(np.array(features.todense()))
        features = normalize(features)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = np.genfromtxt("{}{}.label".format(path, dataset),
                               dtype=int)
        # no_class = len(set(labels))
        no_class = set(labels)

        idx = np.arange(len(labels))

        labels = torch.FloatTensor(labels)
        np.random.shuffle(idx)
        idx = np.arange(len(labels))
        # 去掉离群点15个
        empetySet = []
        idx = list(set(idx).difference(empetySet))

        labels = torch.FloatTensor(labels)




        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        np.random.shuffle(idx)

        # print(idx)

        train_size = math.ceil(lr)
        #print(math.ceil(len(labels) * labelRatio / 100 / len(no_class)))

        train_size = [train_size for i in range(len(no_class))]

        idx_train = []
        count = [0 for i in range(len(no_class))]
        label_each_class = train_size
        next = 0


        for i in idx:
            if count == label_each_class:
                break
            next += 1
            for j in range(len(no_class)):

                if j== labels[i] and count[j] < label_each_class[j]:
                    idx_train.append(i)
                    count[j] += 1

        idx_val = idx[next:next + 500]
        test_idx_range = np.loadtxt(path + 'test' + str(60) + ".txt", dtype=int)
        test_size = len(test_idx_range.tolist())
        idx_test = idx[-test_size:] if test_size else idx[next:]


        idx_unlabel = list(set(idx).difference(set(idx_train)))
        # train = np.loadtxt(path + 'train' + str(20) + ".txt", dtype=int)
        #
        # test = np.loadtxt(path + 'test' + str(60) + ".txt", dtype=int)
        # idx_train = train.tolist()
        # idx_test = test.tolist()
        # idx_val = idx[len(idx_train):len(idx_train) + 500]
        #
        # idx_unlabel = list(set(idx).difference(set(idx_train)))
    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []

        for i in range(len(names)):
            with open(path+"ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)

        test_idx_reorder = parse_index_file(path+"ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)
        empetySet = []
        if dataset == 'citeseer':
            #Citeseer dataset contains some isolated nodes in the graph
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            # 找到离群点
            # test_idx_range_reorder = list(test_idx_range_full)
            # empetySet = set(test_idx_range_reorder).difference(set(test_idx_range))

            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended

            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        features = normalize(features)

        features = torch.FloatTensor(np.array(features.todense()))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        no_class = set(np.where(labels)[1])

        def missing_elements(L):
            start, end = L[0], L[-1]
            return sorted(set(range(start, end+1)).difference(L))


        if dataset == 'citeseer':
            save_label = np.where(labels)[1]
            L = np.sort(test_idx_reorder)
            missing = missing_elements(L)

            for element in missing:
                save_label = np.insert(save_label, element, 0)

            labels = torch.FloatTensor(save_label)
        else:
            labels = torch.FloatTensor(np.where(labels)[1])



        idx = np.arange(len(labels))
        # 去掉离群点15个
        idx = list(set(idx).difference(empetySet))

        labels = torch.FloatTensor(labels)




        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        np.random.shuffle(idx)

        # print(idx)

        train_size = math.ceil(lr)
        #print(math.ceil(len(labels) * labelRatio / 100 / len(no_class)))

        train_size = [train_size for i in range(len(no_class))]

        idx_train = []
        count = [0 for i in range(len(no_class))]
        label_each_class = train_size
        next = 0


        for i in idx:
            if count == label_each_class:
                break
            next += 1
            for j in range(len(no_class)):

                if j== labels[i] and count[j] < label_each_class[j]:
                    idx_train.append(i)
                    count[j] += 1

        idx_val = idx[next:next + 500]
        test_size = len(test_idx_range.tolist())
        idx_test = idx[-test_size:] if test_size else idx[next:]


        idx_unlabel = list(set(idx).difference(set(idx_train)))

    # print(len(idx_unlabel))

    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y)+500)
    # idx_test = test_idx_range.tolist()



    # idx_train, idx_val, idx_test, idx_unlabel = list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test,idx_unlabel]))
    return features, labels, idx_train, idx_val, idx_test,idx_unlabel


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()

# def load_data(path="../data/cora/", dataset="cora"):
#     """Load citation network dataset (cora only for now)"""
#     print('Loading {} dataset...'.format(dataset))
#
#     idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
#                                         dtype=np.dtype(str))
#     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     labels = encode_onehot(idx_features_labels[:, -1])
#
#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#     idx_map = {j: i for i, j in enumerate(idx)}
#     edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
#                                     dtype=np.int32)
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=np.int32).reshape(edges_unordered.shape)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(labels.shape[0], labels.shape[0]),
#                         dtype=np.float32)
#
#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#
#     features = normalize(features)
#     adj = normalize(adj + sp.eye(adj.shape[0]))
#
#     idx_train = range(140)
#     idx_val = range(200, 500)
#     idx_test = range(500, 1500)
#
#     features = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(np.where(labels)[1])
#     adj = sparse_mx_to_torch_sparse_tensor(adj)
#
#     idx_train = torch.LongTensor(idx_train)
#     idx_val = torch.LongTensor(idx_val)
#     idx_test = torch.LongTensor(idx_test)
#
#     return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
