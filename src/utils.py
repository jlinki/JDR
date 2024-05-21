import copy
import numpy as np
import torch
import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric.utils import to_dense_adj

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, seed, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing
    torch.manual_seed(seed)
    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data


def plot_class_dist(data):
    import seaborn as sns
    import collections
    labels = data.y.numpy()
    num_labels = len(data.y.unique())
    connected_labels_set = list(map(lambda x: labels[x], data.edge_index.numpy()))
    connected_labels_set = np.array(connected_labels_set)
    edge_attr = np.array(data.edge_attr)
    def add_missing_keys(counter, classes):
        for x in classes:
            if x not in counter.keys():
                counter[x] = 0
        return counter
    label_connection_counts = []
    for i in range(num_labels):
        if data.edge_attr is None:
            connected_labels = connected_labels_set[:, np.where(connected_labels_set[0] == i)[0]]
            counter = collections.Counter(connected_labels[1])
            counter = dict(counter)
            counter = add_missing_keys(counter, range(num_labels))
            items = sorted(counter.items())
            items = [x[1] for x in items]
        else:
            connected_labels_indices = np.where(connected_labels_set[0] == i)[0]
            connected_labels = connected_labels_set[:, connected_labels_indices]
            connected_attr = edge_attr[connected_labels_indices]
            items = []
            for j in range(num_labels):
                indices = np.where(connected_labels[1] == j)[0]
                items.append(np.sum(connected_attr[indices]))
        label_connection_counts.append(items)
    label_connection_counts = np.array(label_connection_counts)
    plt.figure(figsize=(9, 7))
    plt.rcParams["font.size"] = 13
    hm = sns.heatmap(label_connection_counts, annot=True, cmap='hot_r', cbar=True, square=True)
    plt.xlabel("class",size=20)
    plt.ylabel("class",size=20)
    plt.tight_layout()
    plt.show()

    def scaling(array):
        return array / sum(array)

    label_connection_counts_scaled = np.apply_along_axis(scaling, 1, label_connection_counts)
    plt.figure(figsize=(9, 7))
    plt.rcParams["font.size"] = 13
    hm = sns.heatmap(
        label_connection_counts_scaled,
        annot=True,
        cmap='hot_r',
        fmt="1.2f",
        cbar=True,
        square=True)
    plt.xlabel("class", size=20)
    plt.ylabel("class", size=20)
    plt.tight_layout()
    plt.show()


def add_noise_A(data, args):
    adj = to_dense_adj(data.edge_index)[0]
    n = adj.shape[0]
    p = args.noise_A_rate * (adj == 1).sum() / (n**2)
    noise = torch.bernoulli(torch.ones_like(adj) * p)
    upper_triangular = torch.triu(noise, diagonal=0).to(adj.device)
    lower_triangular = upper_triangular.T.fill_diagonal_(0)
    adj_noise = ((adj + upper_triangular + lower_triangular)==1).long()
    data.edge_index = torch_geometric.utils.dense_to_sparse(adj_noise)[0]
    return data


def add_noise_X(data, args):
    if data.x[data.x == 1].sum() == data.x[data.x > 0].sum():
        p = args.noise_X_rate * (data.x == 1).sum() / (data.x.shape[0]*data.x.shape[1])
        noise = torch.bernoulli(torch.ones_like(data.x) * p)
        X = ((data.x + noise) == 1).float()
        data.x = X
    else:
        data_x_std = torch.std(data.x)
        data.x = data.x + torch.randn_like(data.x) * data_x_std * args.noise_X_rate
    return data


def get_component(dataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def update_args_from_yaml(args):
    """
    Update the arguments based on the contents of the yaml file.
    Args:
        args: args from argparse
    Returns: args
    """
    if args.net.lower() in ['gcn', 'gprgnn', 'mlp', 'spectralclustering', 'none']:
        import yaml
        dataset_name = args.dataset.lower()
        name = 'configs/' + dataset_name + '.yaml'
        if args.denoise_default.lower() == "gcn":
            denoise = "_denoise"
            args.denoise = 'Yes'
            if args.dataset.lower() in ['cora', 'citeseer', 'pubmed', 'photo', 'computers']:
                if args.data_split == "dense":
                    denoise += "_dense"
            if args.net == 'none':
                curr_net = 'gcn'
            else:
                curr_net = args.net.lower()

        elif args.denoise_default.lower() == "gprgnn":
            denoise = "_denoise_gprgnn"
            args.denoise = 'Yes'
            if args.dataset.lower() in ['cora', 'citeseer', 'pubmed', 'photo', 'computers']:
                if args.data_split == "dense":
                    denoise += "_dense"
            if args.net == 'none':
                curr_net = 'gcn'
            else:
                curr_net = args.net.lower()
        else:
            denoise = ""
            curr_net = args.net.lower()

        try:
            with open(name, 'r') as f:
                yaml_dict = yaml.safe_load(f)[curr_net + denoise]
            for key in yaml_dict:
                if key in args:
                    if key in ['rewired_ratio_A', 'rewired_ratio_X', 'rewired_ratio_X_non_binary']:
                        setattr(args, key, round(yaml_dict[key], args.yaml_digits))
                    else:
                        setattr(args, key, yaml_dict[key])
        except KeyError:
            print("[WARNING] Combination not in config file, using default parameters instead")
        except TypeError:
            print("[WARNING] Combination not in config file, using default parameters instead")
        except FileNotFoundError:
            print("[WARNING] Combination not in config file, using default parameters instead")

        if args.rewire_default in ['borf', 'borf_gprgnn']:
            rewire = "_" + args.rewire_default
            args.rewire = 'borf'
            if args.dataset.lower() in ['cora', 'citeseer', 'pubmed', 'photo', 'computers']:
                if args.data_split == "dense":
                    rewire += "_dense"
            try:
                with open(name, 'r') as f:
                    yaml_dict = yaml.safe_load(f)['rewire' + rewire]
                for key in yaml_dict:
                    if key in args:
                        setattr(args, key, yaml_dict[key])
            except KeyError:
                print("[WARNING] Combination for borf not in config file, using default parameters instead")

        if args.rewire_default in ['ppr']:
            rewire = "_" + args.rewire_default
            if args.net.lower() in ["gcn", "gprgnn"]:
                net = args.net.lower()
            else:
                net = "gcn"
            rewire += '_' + net
            args.rewire = 'ppr'
            if args.denoise_default.lower() in ["gcn", "gprgnn"]:
                denoise = "_denoise"
                args.denoise = 'Yes'
            else:
                denoise = ""
            if args.dataset.lower() in ['cora', 'citeseer', 'pubmed', 'photo', 'computers']:
                if args.data_split == "dense":
                    rewire += "_dense"
            try:
                with open(name, 'r') as f:
                    yaml_dict = yaml.safe_load(f)['rewire' + rewire + denoise]
                for key in yaml_dict:
                    if key in args:
                        setattr(args, key, yaml_dict[key])
            except KeyError:
                print("[WARNING] Combination for DIGL not in config file, using default parameters instead")
    return args


def random_sort_nodes(data):
    # This mainly relevant for two-class or cSBM, because they are ordered and then the adjacency is a block matrix,
    # which might be easier to optimize/regularize
    print("[INFO] Using random sorting of the nodes")
    bad_sorting = True
    while bad_sorting:
        indices_sort = torch.randperm(len(data.y))
        data2 = copy.deepcopy(data)
        data2.y = data2.y[indices_sort]
        data2.x = data2.x[indices_sort, :]
        data_adj_orig = torch_geometric.utils.to_dense_adj(edge_index=data.edge_index,
                                                           edge_attr=data.edge_attr).squeeze()
        data2_adj = data_adj_orig[indices_sort][:, indices_sort]
        data2.edge_index = torch_geometric.utils.dense_to_sparse(data2_adj)[0]
        if hasattr(data2, 'train_mask'):
            data2.train_mask = data2.train_mask[indices_sort]
        if hasattr(data2, 'val_mask'):
            data2.val_mask = data2.val_mask[indices_sort]
        if hasattr(data2, 'test_mask'):
            data2.test_mask = data2.test_mask[indices_sort]
        A_temp = torch_geometric.utils.to_dense_adj(data2.edge_index).squeeze()
        if A_temp.shape != data_adj_orig.shape:
            continue
        else:
            bad_sorting = False
    return data2


def convert_to_two_class(data, args):
    """
    Convert the dataset to a two class dataset using the classes X and Y from the arg (a_X_b_Y).
    Args:
        data: PyG Data object

    Returns: PyG Data object
    """
    import copy
    import re
    pattern = r'a(\d+)_b(\d+)'
    matches = re.findall(pattern, args.two_class)
    if matches:
        first_class = int(matches[0][0])
        second_class = int(matches[0][1])
    else:
        print("Using default two class setting 0 and 1")
        first_class = 0
        second_class = 1
    # convert to a two class dataset
    indices = torch.where(data.y == first_class)[0]
    indices = torch.cat((indices, torch.where(data.y == second_class)[0]))
    data2 = copy.deepcopy(data)
    data2.y = data2.y[indices]
    indices_sort = data2.y.sort()[1]
    data2.y = (data2.y[indices_sort] == second_class).long()
    data2.x = data2.x[indices, :]
    data2.x = data2.x[indices_sort, :]
    data_adj_orig = torch_geometric.utils.to_dense_adj(edge_index=data.edge_index,
                                                       edge_attr=data.edge_attr).squeeze()
    data2_adj = data_adj_orig[indices, :]
    data2_adj = data2_adj[:, indices]
    data2_adj = data2_adj[indices_sort, :]
    data2_adj = data2_adj[:, indices_sort]
    data2.edge_index = torch_geometric.utils.dense_to_sparse(data2_adj)[0]
    data2.train_mask = data2.train_mask[indices]
    data2.train_mask = data2.train_mask[indices_sort]
    data2.val_mask = data2.val_mask[indices]
    data2.val_mask = data2.val_mask[indices_sort]
    data2.test_mask = data2.test_mask[indices]
    data2.test_mask = data2.test_mask[indices_sort]
    data2.Lambda = 0.5
    data2.n = len(data2.y)
    return data2

def compute_alignment(X: torch.tensor, A: torch.tensor, args, ord=2):
    """
    Compute the alignment between the node features X and the adjacency matrix A.
    We use the min between L_A and L_X to compute the alignment.
    Args:
        X: feature matrix (N, F)
        A: adjacency matrix (N, N)
        args: arguments from argparse
        ord: e.g. 2 or 'fro' (default 2)

    Returns: alignment value (float)
    """
    VX, s, U = torch.linalg.svd(X)
    la, VA = torch.linalg.eigh(A)
    if args.abs_ordering:
        sort_idx = torch.argsort(torch.abs(la))
        VA = VA[:, sort_idx]
    else:
        pass
    rewired_index = min(args.rewired_index_X, args.rewired_index_A)
    VX = VX[:, :rewired_index]
    if args.denoise_offset == 1:
        VA = VA[:, -rewired_index-1:-1]
    else:
        VA = VA[:, -rewired_index:]
    alignment = torch.linalg.norm(torch.matmul(VX.T, VA), ord=ord)
    return alignment.item()