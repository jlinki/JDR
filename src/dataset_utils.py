import torch

import os
import os.path as osp
import pickle
import torch_geometric.transforms as T

from cSBM_dataset import dataset_ContextualSBM
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Actor
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.datasets import LINKXDataset
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected


class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='data/', name=None,
                 p2raw=None,
                 train_percent=0.01,
                 transform=None, pre_transform=None):

        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class WebKB(InMemoryDataset):
    r"""The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def DataLoader(name, normalize_data):
    if 'cSBM' in name:
        path = '../data/'
        dataset = dataset_ContextualSBM(path, name=name)
        #dataset = dataset_ContextualSBM(path, name=name, n=5000, d=5, p=1000, Lambda=None, mu=None, epsilon=3.25, theta=1.0, train_percent=0.2, transform=None, pre_transform=None)
        return dataset, dataset[0]
    else:
        name = name.lower()

    if name in ['cora', 'citeseer', 'pubmed']:
        root_path = '../'
        path = osp.join(root_path, 'data', name)
        if normalize_data:
            dataset = Planetoid(path, name, transform=T.NormalizeFeatures())
        else:
            dataset = Planetoid(path, name)
    elif name in ['computers', 'photo']:
        root_path = '../'
        path = osp.join(root_path, 'data', name)
        if normalize_data:
            dataset = Amazon(path, name, T.NormalizeFeatures())
        else:
            dataset = Amazon(path, name)
    elif name in ['chameleon', 'squirrel']:
        # use everything from "geom_gcn_preprocess=False" and
        # only the node label y from "geom_gcn_preprocess=True"
        if normalize_data:
            preProcDs = WikipediaNetwork(
                root='../data/', name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
            dataset = WikipediaNetwork(
                root='../data/', name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        else:
            preProcDs = WikipediaNetwork(
                root='../data/', name=name, geom_gcn_preprocess=False)
            dataset = WikipediaNetwork(
                root='../data/', name=name, geom_gcn_preprocess=True)
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
        return dataset, data

    elif name in ['film']:
        if normalize_data:
            dataset = Actor(
                root='../data/film', transform=T.NormalizeFeatures())
        else:
            dataset = Actor(root='../data/film')
    elif name in ['texas', 'cornell']:
        if normalize_data:
            dataset = WebKB(root='../data/',
                            name=name, transform=T.NormalizeFeatures())
        else:
            dataset = WebKB(root='../data/', name=name)
    elif name == 'penn94':
        dataset = LINKXDataset(root='../data/', name='penn94')
    elif name == 'twitch-gamers':
        dataset = TwitchGamers(root='../data/')
        data = dataset.data
        dataset.__num_classes__ = data.y.unique().shape[0]
        data.train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        dataset.data = data
    elif name in ['roman-empire', 'amazon-ratings', "minesweeper", "tolokers", "questions"]:
        if normalize_data:
            dataset = HeterophilousGraphDataset(root='../data/', name=name, transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures()]))
        else:
         dataset = HeterophilousGraphDataset(root='../data/', name=name, transform=T.ToUndirected())
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset, dataset[0]


class TwitchGamers(InMemoryDataset):
    def __init__(self, root, name='twitch-gamers', transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['twitch-gamers']

        super(TwitchGamers, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['large_twitch_edges.csv', 'large_twitch_features.csv']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        import pandas as pd
        edges = pd.read_csv('../data/twitch-gamers/' + 'large_twitch_edges.csv')
        nodes = pd.read_csv('../data/twitch-gamers/' + 'large_twitch_features.csv')
        edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
        edge_index = to_undirected(edge_index)
        label, features = load_twitch_gamer(nodes, "mature")
        node_feat = torch.tensor(features, dtype=torch.float)
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
        data = Data(x=node_feat, edge_index=edge_index, y=torch.tensor(label))
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding

    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()

    return label, features