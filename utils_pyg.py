import torch
import os
import torch.nn as nn
import json
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import os.path as osp
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_scipy_sparse_matrix, degree, from_networkx, to_dense_adj
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from torch_geometric.utils import scatter
from torch.utils.data import Dataset
import random


class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'adj': item['adj'],
            'feats': item['x'].clone(),
            'label': item['y'],
            'num_nodes': item['adj'].shape[0],
            'adj_label': item['adj']
        }


class OneHotDegreeOnly(T.BaseTransform):
    def __init__(self, max_degree=None):
        self.max_degree = max_degree
    def fit(self, dataset):
        if self.max_degree is None:
            max_degree = 0
            for data in dataset:
                if data.edge_index.size(1) > 0:
                    row = data.edge_index[0]
                    deg = degree(row, data.num_nodes, dtype=torch.long)
                    max_degree = max(max_degree, deg.max().item())
            self.max_degree = max(0, max_degree)

    def __call__(self, data):
        if data.edge_index.size(1) == 0:
            deg = torch.zeros(data.num_nodes, dtype=torch.long)
        else:
            row = data.edge_index[0]
            deg = degree(row, data.num_nodes, dtype=torch.long)

        deg = torch.clamp(deg, min=0, max=self.max_degree)
        one_hot = torch.zeros(data.num_nodes, self.max_degree + 1)
        one_hot[torch.arange(data.num_nodes), deg] = 1
        data.x = one_hot.float()

        return data

def Dataset(DS, args, is_train=False, need_str_enc=True):
    print('dataset: ', DS)
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)

    if 'Tox21_' not in DS:
        if DS in ['IMDB-BINARY', 'REDDIT-BINARY', 'COLLAB']:
            dataset = TUDataset(path, name=DS, use_node_attr=True)
            transform = OneHotDegreeOnly(max_degree=10)
            transform.fit(dataset)
            dataset.transform = transform
        else:
            dataset = TUDataset(path, name=DS, use_node_attr=True)
            if DS == 'AIDS':
                dataset.transform = T.remove_isolated_nodes.RemoveIsolatedNodes()

        dataset_num_features = dataset.num_node_features
        datanum = len(dataset)

        max_num_nodes = 0
        for data in dataset:
            num_nodes = data.num_nodes if data.num_nodes is not None else data.x.size(0)
            if num_nodes > max_num_nodes:
                max_num_nodes = num_nodes

        data_list = []
        for data in dataset:
            data.edge_attr = None
            data.num_nodes = None
            data_list.append(data)


        print('Start Structure')
        if need_str_enc:
            data_list = init_structural_encoding(data_list, rw_dim=16, dg_dim=16, is_cat=True)
        str_feat_num = data_list[0].x.size(1)
        print('Start Feature')
        for g in data_list:
            g['max_node'] = max_num_nodes
            feat = g['x']
            feat = preprocess_features(feat)
            feat = feat_alignment(feat, args.input_dim, max_num_nodes)
            g['x'] = feat
        print('data_num: ', datanum, 'orig_feat_num:', dataset_num_features, 'str_feat_num:', str_feat_num, 'max_node:', max_num_nodes ,'input_size', feat.size())

        for g in data_list:
            g['y'] = 1 if g['y'] == 0 else 0

        if is_train == False:
            if DS in ['DD']:
                batch_size_test = 10
            elif DS in ['REDDIT-BINARY']:
                batch_size_test = 40
            else:
                batch_size_test = args.batch_size_test

            data_loader = create_combined_dataloader(data_list, batch_size_test, args.shot)


        else:
            graph_train = []
            for g in data_list:
                if g['y'] == 0:
                    graph_train.append(g)
            data_loader = DataLoader(graph_train, batch_size=args.batch_size, shuffle=True)

    elif 'Tox21_' in DS:

        data_train = TUDataset(path, name=DS + '_training', use_node_attr=True)
        max_num_nodes = 0
        for data in data_train:
            num_nodes = data.num_nodes if data.num_nodes is not None else data.x.size(0)
            if num_nodes > max_num_nodes:
                max_num_nodes = num_nodes

        data_list = []
        for data in data_train:
            data.edge_attr = None
            data.num_nodes = None
            data_list.append(data)
        datanum = len(data_list)
        dataset_num_features = data_train.num_node_features

        print('Start Structure')
        if need_str_enc:
            data_list = init_structural_encoding(data_list, rw_dim=16, dg_dim=16, is_cat=True)
        str_feat_num = data_list[0].x.size(1)
        print('Start Feature')
        for g in data_list:
            g['max_node'] = max_num_nodes
            feat = g['x']
            feat = preprocess_features(feat)
            feat = feat_alignment(feat, args.input_dim, max_num_nodes)
            g['x'] = feat
        print('data_num: ', datanum, 'orig_feat_num:', dataset_num_features, 'str_feat_num:', str_feat_num, 'max_node:',
              max_num_nodes, 'input_size', feat.size())

        for g in data_list:
            g['y'] = 0 if g['y'] == 0 else 1

        if is_train == False:
            data_loader = create_combined_dataloader(data_list, args.batch_size_test, args.shot)

        else:
            graph_train = []
            for g in data_list:
                if g['y'] == 0:
                    graph_train.append(g)
            data_loader = DataLoader(graph_train, batch_size=args.batch_size, shuffle=True)

    return data_loader



def edge_index_to_adj(data, max_nodes):
    adj = to_dense_adj(data.edge_index, max_num_nodes=max_nodes)
    data['adj'] = adj.squeeze(0)
    return data

def edge_index_to_adj_batch(data, max_nodes):
    adj = to_dense_adj(
        edge_index=data.edge_index,
        batch=data.batch,
        max_num_nodes=max_nodes
    )
    eye = torch.eye(max_nodes, device=adj.device).unsqueeze(0)
    adj = adj + eye
    data.adj = adj
    return data


def preprocess_features(features):
    """Row-normalize feature matrix and convert to dense format"""
    features = sp.lil_matrix(features)
    rowsum = np.array(features.sum(1))
    rowsum[rowsum == 0] = np.finfo(float).eps  # Set zero rowsum to a very small value to avoid divide by zero
    r_inv = np.power(rowsum, -1).flatten()
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    features = np.array(features.todense())
    return features


def expand(X, dims, max_num_nodes):
    X = torch.FloatTensor(X)
    N, C = X.shape
    num_new_features = max_num_nodes - N
    new_feature = torch.zeros(num_new_features, C)
    expanded_X = torch.cat((X, new_feature), dim=0)

    return expanded_X

def feat_alignment(features, dims, max_num_nodes):
    transformer = GaussianRandomProjection(n_components=dims, random_state=0)
    features = transformer.fit_transform(features)
    features = expand(features, dims, max_num_nodes)
    return features


def init_structural_encoding(gs, rw_dim=16, dg_dim=16, is_cat=True):
    for g in gs:
        A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
        D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

        Dinv = sp.diags(D)
        RW = A @ Dinv
        M = RW

        RWSE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(rw_dim - 1):
            M_power = M_power @ M
            RWSE.append(torch.from_numpy(M_power.diagonal()).float())
        RWSE = torch.stack(RWSE, dim=-1)

        g_dg = torch.LongTensor(degree(g.edge_index[0], num_nodes=g.num_nodes).numpy())
        DGSE = torch.zeros([g.num_nodes, dg_dim], dtype=torch.float32)
        DGSE.scatter_(1, g_dg.clip(0, dg_dim - 1).view(-1, 1), 1)

        g['x_s'] = torch.cat([RWSE, DGSE], dim=1)
        if is_cat:
            g['x'] = torch.cat([g['x'], g['x_s']], dim=1)
        else:
            g['x'] = g['x_s']
    return gs

class CombinedDataLoader:
    def __init__(self, remaining_loader, few_shot_samples):
        self.remaining_loader = remaining_loader
        self.few_shot_samples = few_shot_samples

    def __iter__(self):
        self.remaining_iterator = iter(self.remaining_loader)
        return self

    def __next__(self):
        remaining_batch = next(self.remaining_iterator)
        few_shot_batch = Batch.from_data_list(self.few_shot_samples)
        combined_batch = Batch.from_data_list([remaining_batch] + [few_shot_batch])

        return combined_batch

    def __len__(self):
        return len(self.remaining_loader)

def create_combined_dataloader(data_list, batch_size, few_shot_count, shuffle=False):
    normal_pool = [data for data in data_list if data.y == 0]
    normal_indices = np.random.choice(len(normal_pool), size=few_shot_count, replace=False)
    selected_normals = [normal_pool[i] for i in normal_indices]

    few_shot_samples = selected_anomalies + selected_normals
    np.random.shuffle(few_shot_samples)

    few_shot_ids = {id(data) for data in few_shot_samples}
    remaining_samples = [data for data in data_list if id(data) not in few_shot_ids]

    remaining_loader = DataLoader(remaining_samples, batch_size=batch_size, shuffle=shuffle)

    return CombinedDataLoader(remaining_loader, few_shot_samples)



def loss_func_adj(adj, A_hat):
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sum(diff_structure, -1)
    structure_cost = torch.mean(structure_reconstruction_errors, 1)
    return structure_reconstruction_errors, structure_cost

def loss_func_feat(attrs, X_hat, batch):
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sum(diff_attribute, -1)
    attribute_cost = scatter(attribute_reconstruction_errors, batch, dim=0, reduce='mean')
    return attribute_reconstruction_errors, attribute_cost


def test_eval(labels_list, probs_list):
    score = {}
    with torch.no_grad():
        labels = torch.cat(labels_list, dim=0)
        probs = torch.cat(probs_list, dim=0)

        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(probs):
            probs = probs.cpu().numpy()

        score['AUROC'] = roc_auc_score(labels, probs)
        score['AP'] = average_precision_score(labels, probs)

        idx = probs.argsort()
        pred_label = labels[idx]
        k = 100
        predk = pred_label[-k:]
        score['Precision@K'] = sum(predk) / len(predk)
        score['Recall@K'] = sum(predk) / sum(pred_label)

    return score


def read_json(model, shot, json_dir):
    # Construct the filename based on the dataset name and shot
    filename = f"{json_dir}/{model}_{shot}.json"

    # Check if the file exists
    if os.path.exists(filename):
        # Read the JSON file and return the dictionary
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
                return data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file {filename}: {e}")
                return None
    else:
        print(f"JSON file {filename} not found.")
        return None