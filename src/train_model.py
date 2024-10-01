from dataset_utils import DataLoader
from utils import *
from GNN_models import *
from denoise_jointly import denoise_jointly, denoise_jointly_large

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch_geometric
import copy
import numpy as np
import scipy.stats as stats
from torch_geometric.utils import to_dense_adj
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score


def RunExp(exp_i, args, dataset, data, Net, percls_trn, val_lb):

    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        if args.dataset.lower() in ['minesweeper', 'tolokers', 'questions']:
            loss = F.binary_cross_entropy_with_logits(out, data.y[data.train_mask].float())
        else:
            loss = F.nll_loss(out, data.y[data.train_mask])
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        loss_name = ["train", "val", "test"]
        for index, (_, mask) in enumerate(data('train_mask', 'val_mask', 'test_mask')):
            if args.dataset.lower() in ["minesweeper", "tolokers", "questions"]:
                pred = logits[mask].squeeze()
                acc = roc_auc_score(data.y[mask].cpu(), pred.detach().cpu())
                loss = F.binary_cross_entropy_with_logits(logits[mask].squeeze(), data.y[mask].float())
            else:
                pred = logits[mask].max(1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                loss = F.nll_loss(model(data)[mask], data.y[mask])
            if args.wandb_log:
                if loss_name[index] != "test":
                    wandb.log({loss_name[index] + "_loss": loss, loss_name[index] + "_acc": acc * 100, "epoch": epoch})
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    appnp_net = Net(dataset, args)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.dataset.lower() in ['twitch-gamers']:
        data.train_mask, data.val_mask, data.test_mask = rand_train_test_idx(
            data, train_prop=args.train_rate, valid_prop=args.val_rate, curr_seed=exp_i + args.run_num)

    elif args.dataset.lower() in ['penn94']:
        if args.original_split:
            if exp_i == 0:
                data.train_mask_arr = copy.deepcopy(dataset.data.train_mask)
                data.val_mask_arr = copy.deepcopy(dataset.data.val_mask)
                data.test_mask_arr = copy.deepcopy(dataset.data.test_mask)
            data.train_mask = data.train_mask_arr[:, exp_i % 5]
            data.val_mask = data.val_mask_arr[:, exp_i % 5]
            data.test_mask = data.test_mask_arr[:, exp_i % 5]
        else:
            data.train_mask, data.val_mask, data.test_mask = rand_train_test_idx(
                data, train_prop=args.train_rate, valid_prop=args.val_rate, curr_seed=exp_i + args.run_num)

    elif args.dataset.lower() in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        if args.original_split:
            if exp_i == 0:
                data.train_mask_arr = copy.deepcopy(dataset.data.train_mask)
                data.val_mask_arr = copy.deepcopy(dataset.data.val_mask)
                data.test_mask_arr = copy.deepcopy(dataset.data.test_mask)
            data.train_mask = data.train_mask_arr[:, exp_i % 10]
            data.val_mask = data.val_mask_arr[:, exp_i % 10]
            data.test_mask = data.test_mask_arr[:, exp_i % 10]
        else:
            permute_masks = random_planetoid_splits
            data = permute_masks(data, dataset.num_classes, exp_i + args.run_num, percls_trn, val_lb)
    else:
        permute_masks = random_planetoid_splits
        data = permute_masks(data, dataset.num_classes, exp_i + args.run_num, percls_trn, val_lb)

    model, data = appnp_net.to(device), data.to(device)

    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
    elif args.net in ['GCN']:
        if args.weight_decay_type == "digl":
            optimizer = torch.optim.Adam([{
                'params': model.conv1.parameters(),
                'weight_decay': args.weight_decay
            },
                {
                'params': model.conv2.parameters(),
                'weight_decay': 0
            }
            ],
                lr=args.lr)
        elif args.weight_decay_type == "gprgnn":
            optimizer = torch.optim.Adam(model.parameters(),
                                         weight_decay=args.weight_decay,
                                         lr=args.lr
                                         )
    elif args.net in ['MLP']:
        if args.weight_decay_type == "digl":
            optimizer = torch.optim.Adam([{
                'params': next(model.children()).parameters(),
                'weight_decay': args.weight_decay
            },
            ],
                lr=args.lr)
        elif args.weight_decay_type == "gprgnn":
            optimizer = torch.optim.Adam(model.parameters(),
                                         weight_decay=args.weight_decay,
                                         lr=args.lr
                                         )
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    counter = 0

    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if (args.early_stop_loss == 'acc' and (val_acc > best_val_acc)):
            best_val_acc = val_acc
            best_val_loss = val_loss
            if args.wandb_log:
                wandb.run.summary["best_val_acc"] = val_acc * 100
                wandb.run.summary["best_val_loss"] = val_loss
            test_acc = tmp_test_acc
            if args.wandb_log:
                wandb.run.summary["test_acc"] = test_acc * 100
            if args.net == 'GPRGNN':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha
            counter = 0

        elif (args.early_stop_loss == 'loss' and (val_loss < best_val_loss)):
            best_val_acc = val_acc
            best_val_loss = val_loss
            if args.wandb_log:
                wandb.run.summary["best_val_acc"] = val_acc * 100
                wandb.run.summary["best_val_loss"] = val_loss
            test_acc = tmp_test_acc
            if args.wandb_log:
                wandb.run.summary["test_acc"] = test_acc * 100
            if args.net == 'GPRGNN':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha
            counter = 0

        else:
            if args.early_stop_type == 'digl':
                counter += 1
                if counter == args.early_stopping:
                    break

        if (epoch >= 0 and args.early_stop_type == 'gprgnn'):
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

    return test_acc, best_val_acc, Gamma_0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Hyperparameters for training
    parser.add_argument('--net', type=str,
                        choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN', 'MLP', 'SpectralClustering', 'none'],
                        default='GCN', help="Choose the GNN model to train")
    parser.add_argument('--epochs', type=int, default=10000,
                        help="Number of epochs to train (early stopping might change this")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for training")
    parser.add_argument('--weight_decay', type=float, default=0.0005, help="Weight decay for training")
    parser.add_argument('--weight_decay_type', type=str, default='gprgnn', choices=['digl', 'gprgnn'],
                        help="Type of weight decay first layer or all layers (only for GCN)")
    parser.add_argument('--early_stopping', type=int, default=200, help="Number of epochs to wait for early stopping")
    parser.add_argument('--early_stop_type', type=str, default='gprgnn', choices=['digl', 'gprgnn'],
                        help="Type of early stopping: mean (gprgnn) or just wait (digl)")
    parser.add_argument('--early_stop_loss', type=str, default='loss', choices=['acc', 'loss'],
                        help="Early stopping based on loss or accuracy")
    parser.add_argument('--hidden', type=int, default=64, help="Number of hidden units in the GNN")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate for training")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in the GNN")
    parser.add_argument('--data_split', default='sparse', choices=['sparse', 'dense', 'gdl', 'sparse5', 'half'],
                        help="sparse means 0.025 for train/val, dense means 0.6/0.2")
    parser.add_argument('--set_seed', default=True, action=argparse.BooleanOptionalAction,
                        help="Set seed for reproducibility")
    parser.add_argument('--original_split', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="Use original split from dataset")

    # GPRGNN specific hyperparameters
    parser.add_argument('--K', type=int, default=10, help="Number of hops (filter order) for GPRGNN")
    parser.add_argument('--alpha', type=float, default=0.1, help="Alpha for GPRGNN Initialization")
    parser.add_argument('--dprate', type=float, default=0.5, help="Dropout rate for GPRGNN")
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR', help="Initialization for GPRGNN")
    parser.add_argument('--ppnp', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'], help="Choose the propagation method for GPRGNN")

    # GAT specific hyperparameters
    parser.add_argument('--heads', default=8, type=int, help="Number of attention heads for GAT")
    parser.add_argument('--output_heads', default=1, type=int, help="Number of output heads for GAT")

    # Hyperparameters of data
    parser.add_argument('--dataset', default='Cora', help="Dataset to use")
    parser.add_argument('--cuda', type=int, default=0, help="Which GPU to use")
    parser.add_argument('--RPMAX', type=int, default=100, help="Number of experiments to run (different seeds)")
    parser.add_argument('--run_num', type=int, default=0, help="Starting run number also first seed")
    parser.add_argument('--use_yaml', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="Use yaml file for default parameters")
    parser.add_argument('--normalize_data', type=str,
                        choices=['Yes', 'No'],
                        default='No', help="Normalize the node features before training")
    parser.add_argument('--use_lcc', type=str,
                        choices=['Yes', 'No'],
                        default='No', help="Use only the largest connected component")
    parser.add_argument('--random_sort', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="Randomly sort the nodes, old, kept for reproducibility")
    parser.add_argument('--two_class', type=str, default='None', help="Convert the dataset to two classes")

    # Parameters for joint denoising
    parser.add_argument('--denoise', type=str,
                        choices=['Yes', 'No'],
                        default='No', help="Use a denoisng method")
    parser.add_argument('--denoise_type', type=str,
                        choices=['jointly'],
                        default='jointly', help="Type of denoising, only jointly implemented")
    parser.add_argument('--denoise_A', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="If denoise, denoise the adjacency matrix")
    parser.add_argument('--denoise_x', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="If denoise, denoise the node features")
    # Main hyperparemeters for denoising
    parser.add_argument('--rewired_index_X', type=int, default=100,
                        help="Number of singular vectors to use when denoising X")
    parser.add_argument('--rewired_index_A', type=int, default=100,
                        help="Number of eigenvectors to use when denoising A")
    parser.add_argument('--rewired_ratio_X', type=float, default=0.1,
                        help="Interpolation Ratio: ratio of new singular vectors to use when denoising X")
    parser.add_argument('--rewired_ratio_A', type=float, default=0.1,
                        help="Interpolation Ratio: ratio of new eigenvectors to use when denoising A")
    parser.add_argument('--rewired_ratio_X_non_binary', type=float, default=0.1,
                        help="Interpolation Ratio: ratio of new X to use when denoising X")
    parser.add_argument('--denoise_iterations', type=int, default=10, help="Number of iterations for denoising")
    parser.add_argument('--denoise_A_k', type=int, default=64, help="Number of entries to keep in A")
    parser.add_argument('--abs_ordering', type=str,
                        choices=['Yes', 'No'],
                        default='No', help= "Activate for heterophilic datasets")
    parser.add_argument('--denoise_offset', type=int, default=0, help="Offset for eigenvectors in A for denoising")
    parser.add_argument('--denoise_default', type=str,
                        choices=['GCN', 'GPRGNN', 'gcn', 'gprgnn', 'No'],
                        default='No', help="Use default parameters for denoising from yaml")
    # Experimental hyperparameters
    parser.add_argument('--use_edge_attr', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="Build a weighted graph with 1d edge features (weights)")
    parser.add_argument('--denoise_A_eps', type=float,
                        default=0.01,
                        help="Threshold for sparsifying A, set A_k to zero to use this")
    parser.add_argument('--use_node_attr', type=str,
                        choices=['Yes', 'No'],
                        default='No', help="Option to sparsify the node features")
    parser.add_argument('--denoise_X_k', type=int, default=64, help="Number of largest entries in X to keep")
    parser.add_argument('--denoise_X_eps', type=float, default=0.01, help="Threshold for sparsifying X")
    parser.add_argument('--use_right_eigvec', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="When multipling back X use the original right singluar vectors")
    # binary A and X
    parser.add_argument('--denoise_non_binary', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="Don't binarize the adjacency matrix and node features")
    parser.add_argument('--flip_number_X_1', type=int, default=1000, help="Number of entries to flip in X from 1 to 0")
    parser.add_argument('--flip_number_X_0', type=int, default=1000, help="Number of entries to flip in X from 0 to 1")
    parser.add_argument('--flip_number_A_1', type=int, default=100, help="Number of entries to flip in A from 1 to 0")
    parser.add_argument('--flip_number_A_0', type=int, default=100, help="Number of entries to flip in A from 0 to 1")
    parser.add_argument('--normalize_data_after_denoise', type=str,
                        choices=['Yes', 'No'],
                        default='No', help="Normalize the node features after denoising if denoising is binary")
    # Ablations
    parser.add_argument('--yaml_digits', type=int, default=16,
                        help="Number of digits of interpolation ratio in the yaml file")
    parser.add_argument('--rewire_index_offset', type=int, default=0, help="Offset for rewiring index in A and X")
    parser.add_argument('--kernel_X', type=str,
                        choices=['Yes', 'No'],
                        default='No', help="Use a kernelized node features for denoising")

    # Spectral clustering
    parser.add_argument('--sc_gamma', type=float, default=1.0, help="Gamma for spectral clustering")

    # Parameters for noisy data
    parser.add_argument('--noise_X_rate', type=float, default=0.0, help="Rate of noise in the node features")
    parser.add_argument('--noise_A_rate', type=float, default=0.0, help="Rate of noise in the adjacency matrix")

    # other rewire methods
    parser.add_argument('--rewire', type=str, default='none',
                        choices=['none', 'ppr', 'heat', 'fosr', 'borf', 'borf4', 'borf5', 'knn', 'knn_weighted', 'heat_eps', 'ppr_eps'],
                        help="Chosse a rewiring method")
    parser.add_argument('--rewire_default', type=str,
                        choices=['No', 'fosr', 'borf', 'borf_gprgnn', 'ppr'],
                        default='No', help="Use default parameters for rewiring from yaml")
    parser.add_argument('--rewire_knn', type=int,
                        default=64, help="Number of nearest neighbors for knn rewiring")

    # DIGL
    parser.add_argument('--rewire_alpha', type=float, default=0.1, help="Alpha (random teleport probability) for DIGL")
    parser.add_argument('--rewire_k', type=int, default=64, help="Number entries to keep in A for DIGL")
    parser.add_argument('--rewire_t', type=float, default=3.0, help="t for heat kernel in DIGL")
    parser.add_argument('--rewire_eps', type=float, default=0.0001, help="eps for thresholding in DIGL")
    parser.add_argument('--self_loop_weight', type=float, default=None, help="Self loop weight for DIGL")
    parser.add_argument('--normalization_in', type=str, default='sym', help="Normalization in for DIGL")
    parser.add_argument('--normalization_out', type=str, default=None, help="Normalization out for DIGL")

    # BORF and FoSR
    parser.add_argument('--fosr_num_iterations', type=int, default=50, help="Number of rewire iterations for BORF")
    parser.add_argument('--borf_num_iterations', type=int, default=3, help="Number of rewire iterations for BORF")
    parser.add_argument('--borf_batch_add', type=int, default=20,
                        help="Number of edges to add in each iteration for BORF")
    parser.add_argument('--borf_batch_remove', type=int, default=10,
                        help="Number of edges to remove in each iteration for BORF")

    # WandB Hyperparameters
    parser.add_argument("--wandb_log", default=True, action=argparse.BooleanOptionalAction,
                        help="Use WandB for login; default: True")
    parser.add_argument("--wandb_log_figs", default=True, action=argparse.BooleanOptionalAction,
                        help="Use wandb logging for figures")
    parser.add_argument('--show_class_dist', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    args.use_yaml = True if args.use_yaml == "Yes" else False

    # Load default parameters from yaml file
    if args.use_yaml:
        args = update_args_from_yaml(args)

    # Reproducibility
    if args.set_seed:
        torch.manual_seed(args.run_num)
        np.random.seed(args.run_num)

    # Data splits
    if args.data_split == "sparse":
        args.train_rate = 0.025
        args.val_rate = 0.025
    elif args.data_split == "sparse5":
        args.train_rate = 0.05
        args.val_rate = 0.05
    elif args.data_split == "dense":
        args.train_rate = 0.6
        args.val_rate = 0.2
    elif args.data_split == "half":
        args.train_rate = 0.5
        args.val_rate = 0.25

    # wandb
    if args.wandb_log:
        import wandb
        wandb.init(project="JDR", config=args)
        args = argparse.Namespace(**wandb.config)
    if args.use_yaml:
        if args.rewire_index_offset != 0:
            args.rewired_index_A += args.rewire_index_offset
            args.rewired_index_X += args.rewire_index_offset

    # Convert string to boolean (needed for wandb sweeps)
    args.denoise = True if args.denoise == "Yes" else False
    args.denoise_A = True if args.denoise_A == "Yes" else False
    args.denoise_x = True if args.denoise_x == "Yes" else False
    args.normalize_data = True if args.normalize_data == "Yes" else False
    args.random_sort = True if args.random_sort == "Yes" else False
    args.use_lcc = True if args.use_lcc == "Yes" else False
    args.normalize_data_after_denoise = True if args.normalize_data_after_denoise == "Yes" else False
    args.abs_ordering = True if args.abs_ordering == "Yes" else False
    args.kernel_X = True if args.kernel_X == "Yes" else False
    args.use_edge_attr = True if args.use_edge_attr == "Yes" else False
    args.use_node_attr = True if args.use_node_attr == "Yes" else False
    args.denoise_non_binary = True if args.denoise_non_binary == "Yes" else False
    args.use_right_eigvec = True if args.use_right_eigvec == "Yes" else False

    # nets
    gnn_name = args.net
    if gnn_name == 'GCN':
        if args.dataset.lower() in ['twitch-gamers', 'penn94']:
            Net = GCN_large
        elif args.dataset.lower() in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
            Net = HeteroGCN
        else:
            Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name == 'JKNet':
        Net = GCN_JKNet
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN
    elif gnn_name == 'MLP':
        Net = MLP

    dname = args.dataset
    dataset, data = DataLoader(dname, args.normalize_data)
    if args.wandb_log:
        wandb.run.summary["edges/original_data"] = (data.edge_index.shape[1])
        wandb.run.summary["x_entries/original_data"] = (data.x == 1).sum().item()
    if args.noise_X_rate > 0.0  :
        data = add_noise_X(data, args)
        if args.wandb_log:
            wandb.run.summary["x_entries/noisy_data"] = (data.x == 1).sum().item()
    if args.noise_A_rate > 0.0:
        data = add_noise_A(data, args)
        if args.wandb_log:
            wandb.run.summary["edges/noisy_data"] = (data.edge_index.shape[1])

    # Convert datasets to two classes if needed
    if not args.two_class == 'None':
        data = convert_to_two_class(data, args)
        dataset.data = data

    # Random sorting of the nodes
    if args.random_sort and (args.dataset.lower() not in ['twitch-gamers', 'penn94']):
        data = random_sort_nodes(data)
        dataset.data = data

    # Only use the largest connected component if needed
    if args.use_lcc:
        lcc = get_largest_connected_component(dataset)

        x_new = dataset.data.x[lcc]
        y_new = dataset.data.y[lcc]

        row, col = dataset.data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))

        data = torch_geometric.data.Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
        dataset.data = data

    print("H(G): ", torch_geometric.utils.homophily(data.edge_index, data.y))
    RPMAX = args.RPMAX
    Init = args.Init

    Gamma_0 = None
    alpha = args.alpha
    train_rate = args.train_rate
    val_rate = args.val_rate
    percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
    val_lb = int(round(val_rate*len(data.y)))
    TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
    print(f'True Label rate in {args.dataset} based on {args.data_split} splitting: {TrueLBrate}')

    args.C = len(data.y.unique())
    args.Gamma = Gamma_0

    # plotting the class distribution
    if args.show_class_dist:
        data_plot = copy.deepcopy(data)
        data_plot.edge_attr = torch.ones(data_plot.edge_index.shape[1], 1)
        plot_class_dist(data_plot)

    # Use device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Denoising
    if args.denoise:
        data.Lambda = 0.5
        data.n = len(data.y)
        if args.denoise_type == 'jointly':
            if args.rewired_index_A > len(data.y):
                args.rewired_index_A = len(data.y)
                if args.rewired_index_X > len(data.y):
                    args.rewired_index_X = len(data.y)
                print(f"rewire_index is too large, set to the maximum value {args.rewire_index}")
            if args.dataset.lower() in ['questions', 'twitch-gamers', 'penn94']:
                data = denoise_jointly_large(data, args, 'cpu')
            else:
                data = denoise_jointly(data, args, device)
            if args.show_class_dist:
                plot_class_dist(data.cpu())
        else:
            raise NotImplementedError(f"Denoising type {args.denoise_type} not implemented")

    # Rewiring
    if args.rewire == 'ppr':
        adj_pre = to_dense_adj(data.edge_index)[0]
        transform = T.GDC(
            self_loop_weight=args.self_loop_weight,
            normalization_in=args.normalization_in,
            normalization_out=args.normalization_out,
            diffusion_kwargs=dict(method='ppr', alpha=args.rewire_alpha),
            sparsification_kwargs=dict(method='topk', k=args.rewire_k, dim=0),
            exact=True,
        )
        data = transform(data)
        adj_ppr = to_dense_adj(data.edge_index)[0]
        if args.wandb_log:
            wandb.run.summary["edges/original"] = (adj_pre == 1).sum().item()
            wandb.run.summary["edges/rewired"] = (adj_ppr == 1).sum().item()
            wandb.run.summary["edges/add"] = ((adj_pre - adj_ppr) == -1).sum().item()
            wandb.run.summary["edges/remove"] = ((adj_pre - adj_ppr) == 1).sum().item()
    elif args.rewire == ('ppr_eps'):
        adj_pre = to_dense_adj(data.edge_index)[0]
        transform = T.GDC(
            self_loop_weight=args.self_loop_weight,
            normalization_in=args.normalization_in,
            normalization_out=args.normalization_out,
            diffusion_kwargs=dict(method='ppr', alpha=args.rewire_alpha),
            sparsification_kwargs=dict(method='threshold', eps=args.rewire_eps, dim=0),
            exact=True,
        )
        data = transform(data)
        adj_ppr = to_dense_adj(data.edge_index)[0]
        if args.wandb_log:
            wandb.run.summary["edges/original"] = (adj_pre == 1).sum().item()
            wandb.run.summary["edges/rewired"] = (adj_ppr == 1).sum().item()
            wandb.run.summary["edges/add"] = ((adj_pre - adj_ppr) == -1).sum().item()
            wandb.run.summary["edges/remove"] = ((adj_pre - adj_ppr) == 1).sum().item()
    elif args.rewire == 'heat':
        adj_pre = to_dense_adj(data.edge_index)[0]
        transform = T.GDC(
            self_loop_weight=args.self_loop_weight,
            normalization_in=args.normalization_in,
            normalization_out=args.normalization_out,
            diffusion_kwargs=dict(method='heat', t=args.rewire_t),
            sparsification_kwargs=dict(method='topk', k=args.rewire_k, dim=0),
            exact=True,
        )
        data = transform(data)
        adj_heat = to_dense_adj(data.edge_index)[0]
        if args.wandb_log:
            wandb.run.summary["edges/original"] = (adj_pre == 1).sum().item()
            wandb.run.summary["edges/rewired"] = (adj_heat == 1).sum().item()
            wandb.run.summary["edges/add"] = ((adj_pre - adj_heat) == -1).sum().item()
            wandb.run.summary["edges/remove"] = ((adj_pre - adj_heat) == 1).sum().item()
    elif args.rewire == 'heat_eps':
        adj_pre = to_dense_adj(data.edge_index)[0]
        transform = T.GDC(
            self_loop_weight=args.self_loop_weight,
            normalization_in=args.normalization_in,
            normalization_out=args.normalization_out,
            diffusion_kwargs=dict(method='heat', t=args.rewire_t),
            sparsification_kwargs=dict(method='threshold', eps=args.rewire_eps, dim=0),
            exact=True,
        )
        data = transform(data)
        adj_heat = to_dense_adj(data.edge_index)[0]
        if args.wandb_log:
            wandb.run.summary["edges/original"] = (adj_pre == 1).sum().item()
            wandb.run.summary["edges/rewired"] = (adj_heat == 1).sum().item()
            wandb.run.summary["edges/add"] = ((adj_pre - adj_heat) == -1).sum().item()
            wandb.run.summary["edges/remove"] = ((adj_pre - adj_heat) == 1).sum().item()


    elif args.rewire == "borf":
        adj_pre = to_dense_adj(data.edge_index)[0]
        from preprocessing import borf
        print(f"[INFO] BORF hyper-parameter : num_iterations = {args.borf_num_iterations}")
        print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
        print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
        dataset.data.edge_index, dataset.data.edge_type = borf.borf3(dataset.data, loops=args.borf_num_iterations, remove_edges=False, is_undirected=True, batch_add=args.borf_batch_add, batch_remove=args.borf_batch_remove, dataset_name=args.dataset, graph_index=0)
        data = dataset.data
        print(len(dataset.data.edge_type))

        adj_borf = to_dense_adj(data.edge_index)[0]
        if args.wandb_log:
            wandb.run.summary["edges/original"] = (adj_pre == 1).sum().item()
            wandb.run.summary["edges/rewired"] = (adj_borf == 1).sum().item()
            wandb.run.summary["edges/add"] = args.borf_batch_add * args.borf_num_iterations
            wandb.run.summary["edges/remove"] = args.borf_batch_remove * args.borf_num_iterations
        if args.show_class_dist:
            plot_class_dist(data.cpu())

    elif args.rewire == "borf4":
        adj_pre = to_dense_adj(data.edge_index)[0]
        from preprocessing import borf
        print(f"[INFO] BORF hyper-parameter : num_iterations = {args.borf_num_iterations}")
        print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
        print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
        dataset.data.edge_index, dataset.data.edge_type = borf.borf4(dataset.data, loops=args.borf_num_iterations, remove_edges=False, is_undirected=True, batch_add=args.borf_batch_add, batch_remove=args.borf_batch_remove, dataset_name=args.dataset, graph_index=0)
        data = dataset.data
        print(len(dataset.data.edge_type))

        adj_borf = to_dense_adj(data.edge_index)[0]
        if args.wandb_log:
            wandb.run.summary["edges/original"] = (adj_pre == 1).sum().item()
            wandb.run.summary["edges/rewired"] = (adj_borf == 1).sum().item()
            wandb.run.summary["edges/add"] = args.borf_batch_add * args.borf_num_iterations
            wandb.run.summary["edges/remove"] = args.borf_batch_remove * args.borf_num_iterations
        if args.show_class_dist:
            plot_class_dist(data.cpu())

    elif args.rewire == "borf5":
        adj_pre = to_dense_adj(data.edge_index)[0]
        from preprocessing import borf
        print(f"[INFO] BORF hyper-parameter : num_iterations = {args.borf_num_iterations}")
        print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
        print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
        dataset.data.edge_index, dataset.data.edge_type = borf.borf5(dataset.data, loops=args.borf_num_iterations, remove_edges=False, is_undirected=True, batch_add=args.borf_batch_add, batch_remove=args.borf_batch_remove, dataset_name=args.dataset, graph_index=0)
        data = dataset.data
        print(len(dataset.data.edge_type))

        adj_borf = to_dense_adj(data.edge_index)[0]
        if args.wandb_log:
            wandb.run.summary["edges/original"] = (adj_pre == 1).sum().item()
            wandb.run.summary["edges/rewired"] = (adj_borf == 1).sum().item()
            wandb.run.summary["edges/add"] = args.borf_batch_add * args.borf_num_iterations
            wandb.run.summary["edges/remove"] = args.borf_batch_remove * args.borf_num_iterations
        if args.show_class_dist:
            plot_class_dist(data.cpu())
    elif args.rewire == "knn":
        adj_pre = to_dense_adj(data.edge_index)[0]
        from sklearn.neighbors import kneighbors_graph
        A_knn = kneighbors_graph(data.x, args.rewire_knn, mode='connectivity', include_self=False).todense()
        A_knn = torch.tensor(A_knn)
        data.edge_index = torch_geometric.utils.dense_to_sparse(A_knn)[0]
        if args.wandb_log:
            wandb.run.summary["edges/original"] = (adj_pre == 1).sum().item()
            wandb.run.summary["edges/rewired"] = (A_knn == 1).sum().item()
            wandb.run.summary["edges/add"] = ((adj_pre - A_knn) == -1).sum().item()
            wandb.run.summary["edges/remove"] = ((adj_pre - A_knn) == 1).sum().item()
    elif args.rewire == "knn_weighted":
        adj_pre = to_dense_adj(data.edge_index)[0]
        from sklearn.neighbors import kneighbors_graph
        A_knn = kneighbors_graph(data.x, args.rewire_knn, mode='distance', include_self=False).todense()
        A_knn = torch.tensor(A_knn)
        A_knn = A_knn/A_knn.max()
        data.edge_index, data.edge_weight = torch_geometric.utils.dense_to_sparse(A_knn)
        A_knn = torch_geometric.utils.to_dense_adj(data.edge_index)[0]
        if args.wandb_log:
            wandb.run.summary["edges/original"] = (adj_pre == 1).sum().item()
            wandb.run.summary["edges/rewired"] = (A_knn == 1).sum().item()
            wandb.run.summary["edges/add"] = ((adj_pre - A_knn) == -1).sum().item()
            wandb.run.summary["edges/remove"] = ((adj_pre - A_knn) == 1).sum().item()
    elif args.rewire == 'fosr':
        adj_pre = to_dense_adj(data.edge_index)[0]
        from preprocessing import fosr
        edge_index, _, _ = fosr.edge_rewire(dataset.data.edge_index.numpy(), num_iterations=args.fosr_num_iterations)
        dataset.data.edge_index = torch.tensor(edge_index)
        if args.dataset.lower() not in ['questions', 'twitch-gamers', 'penn94']:
            adj_sdrf = to_dense_adj(data.edge_index)[0]
            if args.wandb_log:
                wandb.run.summary["edges/original"] = (adj_pre == 1).sum().item()
                wandb.run.summary["edges/rewired"] = (adj_sdrf == 1).sum().item()
                wandb.run.summary["edges/add"] = ((adj_pre - adj_sdrf) == -1).sum().item()
                wandb.run.summary["edges/remove"] = ((adj_pre - adj_sdrf) == 1).sum().item()
    elif args.rewire == "none":
        pass
    else:
        raise NotImplementedError(f"Rewiring method {args.rewire} not implemented")

    if args.normalize_data_after_denoise:
        transform = torch_geometric.transforms.NormalizeFeatures()
        data = transform(data)

    if args.show_class_dist:
        data_plot = copy.deepcopy(data)
        data_plot.edge_attr = torch.ones(data_plot.edge_index.shape[1], 1)
        plot_class_dist(data_plot)

    print("H(G) after denoise and/or rewire: ", torch_geometric.utils.homophily(data.edge_index, data.y))
    Results0 = []
    if args.net not in ['SpectralClustering', 'none']:
        for RP in tqdm(range(RPMAX), desc='Running Experiments'):
            test_acc, best_val_acc, Gamma_0 = RunExp(RP,
                args, dataset, data, Net, percls_trn, val_lb)
            Results0.append([test_acc, best_val_acc])
            if args.wandb_log:
                wandb.log({"best_val_acc": best_val_acc * 100, "exp": RP})
                mean_acc_val_acc = np.mean(Results0, axis=0)[1] * 100
                wandb.log({"mean_acc_val_acc": mean_acc_val_acc, "exp": RP})

        test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100
        test_acc_std, val_acc_std = np.sqrt(np.var(Results0, axis=0)) * 100
        Results0_test = np.array(Results0)[:, 0] * 100
        Results0_val = np.array(Results0)[:, 1] * 100

        # Compute 95% confidence interval
        res_val = stats.bootstrap((np.array(Results0_val),), np.mean, confidence_level=0.95, n_resamples=1000)
        res_test = stats.bootstrap((np.array(Results0_test),), np.mean, confidence_level=0.95, n_resamples=1000)
        val_ci_95 = np.max(np.abs(np.array([res_val.confidence_interval.high, res_val.confidence_interval.low]) - np.mean(Results0_val)))
        test_ci_95 = np.max(np.abs(np.array([res_test.confidence_interval.high, res_test.confidence_interval.low]) - np.mean(Results0_test)))
        if args.wandb_log:
            wandb.run.summary["RP_test_acc"] = test_acc_mean
            wandb.run.summary["RP_val_acc"] = val_acc_mean
            wandb.run.summary["RP_test_acc_std"] = test_acc_std
            wandb.run.summary["RP_val_acc_std"] = val_acc_std
            wandb.run.summary["RP_test_acc_95_conf"] = test_ci_95
            wandb.run.summary["RP_val_acc_95_conf"] = val_ci_95
        print(f'{gnn_name} on dataset {args.dataset}, in {RPMAX} repeated experiment with denoising ({args.denoise}) and {args.rewire} rewiring:')
        print(
            f'val acc mean = {val_acc_mean:.4f} \t val acc std = {val_acc_std:.4f} \t val acc 95 conf = {val_ci_95:.4f}')
        print(
            f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t test acc 95 conf = {test_ci_95:.4f}')
    elif args.net == 'SpectralClustering':
        # Clustering
        from sklearn.cluster import SpectralClustering
        def get_acc(y, prediction):
            from scipy.optimize import linear_sum_assignment
            from sklearn.metrics import confusion_matrix
            conf_matrix = confusion_matrix(y, prediction)
            row_ind, col_ind = linear_sum_assignment(-conf_matrix.T)
            optimal_assignment = list(zip(row_ind, col_ind))
            correct_assignments = [conf_matrix[j, i] for i, j in optimal_assignment]
            accuracy = sum(correct_assignments) / len(y)
            corrected_labels = []
            return accuracy, corrected_labels, optimal_assignment

        transform = torch_geometric.transforms.NormalizeFeatures()
        data = transform(data)
        permute_masks = random_planetoid_splits
        data = permute_masks(data, dataset.num_classes, args.run_num, percls_trn, val_lb)
        if torch_geometric.utils.isolated.contains_isolated_nodes(data.edge_index):
            transform = torch_geometric.transforms.RemoveIsolatedNodes()
            data = transform(data)
        A = (torch_geometric.utils.to_dense_adj(edge_index=data.edge_index, edge_attr=data.edge_attr)[0]).squeeze()
        clustering_kern_A_hom = SpectralClustering(n_clusters=dataset.num_classes, assign_labels='kmeans', affinity='precomputed', gamma=args.sc_gamma,
                                             random_state=args.run_num).fit(A.cpu())
        accuracy_A_hom, corrected_labels_A_hom, optimal_assignment_A_hom = get_acc(data.y.cpu(), np.array(clustering_kern_A_hom.labels_))
        accuracy_A_hom_val, corrected_labels_A_hom_val, optimal_assignment_A_hom_val = get_acc(data.y[data['val_mask']].cpu(),
                                                                       np.array(clustering_kern_A_hom.labels_[data['val_mask'].cpu()]))
        clustering_kern_A_het = SpectralClustering(n_clusters=dataset.num_classes, assign_labels='kmeans',
                                               affinity='precomputed', gamma=args.sc_gamma,
                                               random_state=args.run_num).fit(torch.ones_like(A.cpu())-A.cpu())
        accuracy_A_het, corrected_labels_A_het, optimal_assignment_A_het = get_acc(data.y.cpu(),
                                                                                   np.array(clustering_kern_A_het.labels_))
        accuracy_A_het_val, corrected_labels_A_het_val, optimal_assignment_A_het_val = get_acc(data.y[data['val_mask']].cpu(),
                                                                                   np.array(clustering_kern_A_het.labels_[
                                                                                                data['val_mask'].cpu()]))
        clustering_kern_X = SpectralClustering(n_clusters=dataset.num_classes, assign_labels='kmeans', gamma=args.sc_gamma, random_state=args.run_num).fit(data.x.cpu())
        accuracy_X, corrected_labels_X, optimal_assignment_X = get_acc(data.y.cpu(), np.array(clustering_kern_X.labels_))
        accuracy_X_val, corrected_labels_X_val, optimal_assignment_X_val = get_acc(data.y[data['val_mask']].cpu(),
                                                                                   np.array(clustering_kern_X.labels_[
                                                                                                data['val_mask'].cpu()]))
        if args.wandb_log:
            wandb.run.summary["RP_test_acc_A_hom"] = accuracy_A_hom * 100
            wandb.run.summary["RP_test_acc_A_het"] = accuracy_A_het * 100
            wandb.run.summary["RP_test_acc_X"] = accuracy_X * 100
            wandb.run.summary["RP_val_acc_A_hom"] = accuracy_A_hom_val * 100
            wandb.run.summary["RP_val_acc_A_het"] = accuracy_A_het_val * 100
            wandb.run.summary["RP_val_acc_X"] = accuracy_X_val * 100
            if accuracy_A_het > accuracy_A_hom:
                accuracy_A = accuracy_A_het
                accuracy_A_val = accuracy_A_het_val
            else:
                accuracy_A = accuracy_A_hom
                accuracy_A_val = accuracy_A_hom_val
            wandb.run.summary["RP_test_acc_A"] = accuracy_A * 100
            wandb.run.summary["RP_val_acc_A"] = accuracy_A_val * 100
            wandb.run.summary["RP_test_acc"] = np.max(np.array([accuracy_A, accuracy_X])) * 100
            wandb.run.summary["RP_test_acc_mean"] = np.mean(np.array([accuracy_A, accuracy_X])) * 100
            wandb.run.summary["RP_val_acc"] = np.max(np.array([accuracy_A_val,accuracy_X_val])) * 100
            wandb.run.summary["RP_val_acc_mean"] = np.mean(np.array([accuracy_A_val, accuracy_X_val])) * 100
        else:
            print(f'RP_test_acc_A_hom: {accuracy_A_hom * 100:.4f}')
            print(f'RP_test_acc_A_het: {accuracy_A_het * 100:.4f}')
            print(f'RP_test_acc_X: {accuracy_X * 100:.4f}')
            print(f'RP_val_acc_A_hom: {accuracy_A_hom_val * 100:.4f}')
            print(f'RP_val_acc_A_het: {accuracy_A_het_val * 100:.4f}')
            print(f'RP_val_acc_X: {accuracy_X_val * 100:.4f}')
            if accuracy_A_het > accuracy_A_hom:
                accuracy_A = accuracy_A_het
                accuracy_A_val = accuracy_A_het_val
            else:
                accuracy_A = accuracy_A_hom
                accuracy_A_val = accuracy_A_hom_val
            print(f'RP_test_acc_A: {accuracy_A * 100:.4f}')
            print(f'RP_val_acc_A: {accuracy_A_val * 100:.4f}')
            print(f'RP_test_acc: {np.max(np.array([accuracy_A,accuracy_X])) * 100:.4f}')
            print(f'RP_test_acc_mean: {np.mean(np.array([accuracy_A, accuracy_X])) * 100:.4f}')
            print(f'RP_val_acc: {np.max(np.array([accuracy_A_val,accuracy_X_val])) * 100:.4f}')
            print(f'RP_val_acc_mean: {np.mean(np.array([accuracy_A_val, accuracy_X_val])) * 100:.4f}')

    else:
        pass