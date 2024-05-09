import torch
from utilstools.loadAdjMulti import load_adjacency_multiview
import os
import scipy.io
from sklearn.preprocessing import minmax_scale, maxabs_scale, normalize, robust_scale, scale
import numpy as np
import random

def dataprocess(args,dataset_name, k_value, test=-1,load_from_file = True):
    data_dir = os.path.join("./data", dataset_name)
    exist_k = os.path.exists(os.path.join(data_dir, "data_" + str(k_value) + ".pth"))
    if load_from_file and os.path.exists(data_dir) and exist_k:
        print("Loading Existing adjacency matrix.")
        load_data = torch.load(os.path.join(data_dir, "data_" + str(k_value) + ".pth"))
        load_adjs = torch.load(os.path.join(data_dir, "adj_" + str(k_value) + ".pth"))
        features = load_data["features"]
        if dataset_name == "Reuters":
            for idx, feature in enumerate(features[0]):
                features[0][idx] = feature.todense()
        gnd = load_data["gnd"]
        adjs = load_adjs["adjs"].to_dense()
        adj_hats = load_adjs["adj_hats"].to_dense()
        
    else:
        print("Generating adjacency matrix.")
        exist = os.path.exists(os.path.join("./data", f"{dataset_name}.data"))
        if exist:
            datas = torch.load(f"./data/{dataset_name}.data")
            gnd = torch.load(f"./data/{dataset_name}.labels").numpy()
            numpy_arrays = [tensor.numpy() for tensor in datas]

            features = [numpy_arrays]

        else:
            features, gnd = loadMatData(os.path.join("./data", dataset_name))
            features = feature_normalization(features)

        adjs, adj_hats = load_adjacency_multiview(features, k=k_value)
        adjs = torch.from_numpy(adjs)
        adj_hats = torch.from_numpy(adj_hats)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        # save_data = {'features': features, 'gnd': gnd}
        # save_adjs = {'adjs': adjs.to_sparse(), 'adj_hats': adj_hats.to_sparse()}
        #
        # torch.save(save_data, os.path.join(data_dir, "data_" + str(k_value) + ".pth"))
        # torch.save(save_adjs, os.path.join(data_dir, "adj_" + str(k_value) + ".pth"))

        if dataset_name == "Reuters":
            for idx, feature in enumerate(features[0]):
                features[0][idx] = feature.todense()

    p_labeled, p_unlabeled = generate_partition(gnd, args.ratio, args.sf_seed)
    # data_split = torch.load(os.path.join(data_dir, "data_split.pth"))
    # p_labeled = data_split["training"]
    # p_unlabeled = data_split["testing"]

    return features, gnd, p_labeled, p_unlabeled, adjs, adj_hats

def loadMatData(data_name):
    data = scipy.io.loadmat(data_name) 
    features = data['X']

    if data_name.find('3sources')>=0 or data_name.find('WebKB')>=0:
        for i in range(features.shape[1]):
            features[0][i] = features[0][i].astype(np.float32).T
        gnd = data['gt']
    else:
        gnd = data['Y']
    gnd = gnd.flatten()
    gnd = gnd - 1
    return features, gnd

def feature_normalization(features, normalization_type = 'normalize'):
    for idx, fea in enumerate(features[0]):
        if normalization_type == 'minmax_scale':
            features[0][idx] = minmax_scale(fea)
        elif normalization_type == 'maxabs_scale':
            features[0][idx] = maxabs_scale(fea)
        elif normalization_type == 'normalize':
            features[0][idx] = normalize(fea)
        elif normalization_type == 'robust_scale':
            features[0][idx] = robust_scale(fea)
        elif normalization_type == 'scale':
            features[0][idx] = scale(fea)
        elif normalization_type == '255':
            features[0][idx] = np.divide(fea, 255.)
        elif normalization_type == '50':
            features[0][idx] = np.divide(fea, 50.)
        else:
            print("Please enter a correct normalization type!")
    return features


def generate_partition(labels, ratio, seed):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {} ## number of labeled samples for each class
    total_num = round(ratio * len(labels))
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1) # min is 1

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    index = [i for i in range(len(labels))]
    # print(index)
    if seed >= 0:
        random.seed(seed)
        random.shuffle(index)
    labels = labels[index]
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            p_labeled.append(index[idx])
            total_num -= 1
        else:
            p_unlabeled.append(index[idx])
    return p_labeled, p_unlabeled



def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict
