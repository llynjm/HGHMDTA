import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
from torch_geometric.data.batch import Batch
import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
import torch
from torch_geometric.data import Data

import torch


class MoleculeFragData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        # 如果是 cluster_index，偏移量应该是当前分子片段的最大 ID + 1
        if key == 'cluster_index':
            return int(value.max()) + 1
        # 其他属性（如 edge_index, x）遵循默认的偏移逻辑
        # [片段 -> 分子] 的映射索引：由于在 collate 里手动赋予了 batch 内的 i (0~511)
        # 这里必须返回 0，告诉 PyG 不要再额外增加偏移量
        if key == 'frag_graph_edge_index':
            return int(self.cluster_index.max()) + 1
        if key == 'mol_index':
            return 1
        return super().__inc__(key, value, *args, **kwargs)


class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None, smi=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None,
                 target_key=None, target_graph=None):

        # root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.process(xd, xt, smi, y, smile_graph, target_key, target_graph)
        # if os.path.isfile(self.processed_paths[0]):
        #     print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
        #     self.data, self.slices = torch.load(self.processed_paths[0])
        # else:
        #     print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
        #     self.process(xd, xt, smi, y, smile_graph, target_key, target_graph)
        #     self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, smi, y, smile_graph, target_key, target_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_list_2 = []
        data_list_frags = []  # 新增：存储片段数据
        data_len = len(xd)
        for i in range(data_len):
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            drug_smi_label = smi[i]
            target = xt[i]
            labels = y[i]
            key = target_key[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index, fra_edge_index, cluster_index, frag_graph_edge_index = smile_graph[smiles]
            # 计算当前分子的片段数
            num_fragments = int(cluster_index.max()) + 1

            # 初始化 mol_index 为全 0，长度为片段数
            mol_index = torch.zeros((num_fragments,), dtype=torch.long)
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            GCNData.drug_smiles = torch.LongTensor([drug_smi_label])
            # GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            GCNData_frags = MoleculeFragData(x=torch.Tensor(features), frags_edge_index=fra_edge_index,
                                             cluster_index=cluster_index, frag_graph_edge_index=frag_graph_edge_index,
                                             mol_index=mol_index)

            c_size, features, edge_index = target_graph[key]
            GCNData_2 = DATA.Data(x=torch.Tensor(features),
                                  edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                  y=torch.FloatTensor([labels]))
            GCNData_2.target = torch.LongTensor([target])
            GCNData_2.__setitem__('c_size', torch.LongTensor([c_size]))

            # append graph, label and target sequence to data list
            data_list.append(GCNData)
            data_list_frags.append(GCNData_frags)  # 新增
            data_list_2.append(GCNData_2)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
            data_list_2 = [data for data in data_list_2 if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            data_list_2 = [self.pre_transform(data) for data in data_list_2]
        print('Graph construction done. Saving to file.')
        # data, slices = self.collate(data_list)
        # # save preprocessed data:
        # torch.save((data, slices), self.processed_paths[0])
        self.data_mol = data_list
        self.data_frags = data_list_frags  # 新增
        self.data_pro = data_list_2

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_frags[idx], self.data_pro[idx]


def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def collate(data_list):
    """
    data_list 的每个元素是 (__getitem__ 返回的): (mol_data, frags_data, pro_data)
    """
    # 1. 提取三部分数据
    mol_list = [data[0] for data in data_list]
    frags_list = [data[1] for data in data_list]
    pro_list = [data[2] for data in data_list]

    # 2. 为片段数据注入 mol_index (片段 -> 分子的映射)
    for i, frag_data in enumerate(frags_list):
        # 计算当前分子有多少个片段
        # frag_data.batch 存储了原子属于哪个片段，其最大值+1即为片段数
        num_fragments = frag_data.cluster_index.max().item() + 1
        # 创建一个全为 i 的向量，表示这些片段都属于第 i 个分子
        frag_data.mol_index = torch.full((num_fragments,), i, dtype=torch.long)
    # print(frag_data.mol_index[:50])
    # 3. 分别打包成 Batch
    batch_mol = Batch.from_data_list(mol_list)
    batch_frags = Batch.from_data_list(frags_list)
    batch_pro = Batch.from_data_list(pro_list)

    return batch_mol, batch_frags, batch_pro