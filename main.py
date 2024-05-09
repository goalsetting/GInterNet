
from InteractModule_main import GInterNet
from utilstools.LoadData import dataprocess


import argparse
import random

import numpy as np
import yaml
from yaml import SafeLoader
import torch
from ModelMain import trains_multiview, tests_multiview

if __name__ == '__main__':
    torch.cuda.empty_cache()
    datasetStr = "NUS-WIDE"

    #########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=datasetStr)
    parser.add_argument("--sf_seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='herconfig.yaml')
    args = parser.parse_args()
    name = datasetStr

    config = yaml.load(open(args.config), Loader=SafeLoader)[datasetStr]

    torch.manual_seed(config['seed'])
    random.seed(12345)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    k = config['K']
    ratio = config['ratio']
    parser.add_argument("--ratio", type=float, default=ratio, help="label ratio.")
    args = parser.parse_args()

    validate = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #########################################################################

    features, labels, p_labeled, p_unlabeled, adj, adj_nor = dataprocess(args, args.dataset_name, k)
    label = torch.tensor(labels).long().to(device)
    idx_train = torch.tensor(p_labeled).long().to(device)
    idx_test = torch.tensor(p_unlabeled).long().to(device)


    #########################################################################
    data = []
    adjs = []
    for i in range(adj.shape[0]):
        data.append(torch.FloatTensor(features[0][i]).to(device))
        adjs.append(adj_nor[i].float().to(device))
    #########################################################################



    #########################################################################
    index = 1
    lamda = 0
    eita = 0
    acco = np.zeros((11, 11))
    #########################################################################

    for jj in range(11):
        lamda = jj/10
        for jjj in range(11):
            eita = jjj/10
            for i in range(index):
                acc = 0
                torch.cuda.empty_cache()
                model = GInterNet(
                            hid_dim=num_hidden,
                            out_features=int(label.max())+1,
                            features=data,
                            adjs=adjs,
                            label=label,
                            k=2,
                            unlabeled_index=idx_test,
                            dropout=0.5).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                optimizer.zero_grad()
                for epoch in range(1, num_epochs + 1):
                    loss = trains_multiview(model, optimizer, label, idx_train, lamda, eita)
                    realacc = tests_multiview(model, label, idx_test, final=False)
                    if realacc > acc:
                        acc = realacc
                        print("epoch"+str(epoch)+"/"+str(num_epochs)+f", Loss: {loss}")
                        # print(f'LOSSinterGCN{loss}')
                print("=== Final ===interact_gcn")
                print(f"FinalACC:{acc}")
            acco[jj,jjj] = acc
            print(acco)

