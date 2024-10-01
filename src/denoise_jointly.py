from torch_geometric.data import Data
import torch_geometric
import copy
from tqdm import tqdm
import torch


def find_largest(matrix: torch.tensor, k: int) -> torch.tensor:
    """
    Find the k-th largest value in a matrix
    Args:
        matrix: input matrix
        k: number of values to return
    Returns:
    """
    flat_matrix = matrix.flatten()
    values, _ = torch.topk(flat_matrix, k, largest=True, sorted=True)
    return values[-1]


def get_top_k_matrix(matrix: torch.tensor, k: int = 128) -> torch.tensor:
    """
    Get the top k value matrix, all other values are set to 0
    Args:
        matrix: input matrix (N,N)
        k: number of values k to keep

    Returns:
    """
    num_nodes = matrix.shape[0]
    row_idx = torch.arange(num_nodes)
    matrix[matrix.argsort(axis=0)[:num_nodes - k], row_idx] = 0.0
    return matrix


def get_top_k_features(matrix: torch.Tensor, k: int = 128) -> torch.Tensor:
    """
    Get the top k value matrix (rectangular), all other values are set to 0
    Args:
        matrix: input matrix shape (N, F)
        k: number of values k to keep

    Returns:
    """
    _, top_k_indices = matrix.topk(k, dim=1)
    mask = torch.zeros_like(matrix)
    mask.scatter_(1, top_k_indices, 1.)
    matrix *= mask
    return matrix


def get_clipped_matrix(matrix: torch.tensor, eps: float = 0.01) -> torch.tensor:
    """
    Clip the matrix values to 0 if they are below a certain threshold
    Args:
        matrix: input matrix, possibly rectangular
        eps: the threshold value

    Returns:
    """
    matrix[matrix < eps] = 0.0
    return matrix


def get_clipped_features(matrix: torch.tensor, eps: float = 0.01) -> torch.tensor:
    """
        Clip the matrix values to 0 if they are below a certain threshold
        Args:
            matrix: input matrix, possibly rectangular
            eps: the threshold value

        Returns:
    """
    matrix[matrix < eps] = 0.0
    return matrix


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


def denoise_jointly(data, args, device):
    """
    Denoise the graph data jointly using the given arguments
    Args:
        data: PyG data object
        args: arguments from argparse
        device: device to run the operations on (cuda or cpu)

    Returns: denoised PyG data object
    """
    offset = args.denoise_offset
    X = data.x.to(device)
    A = torch_geometric.utils.to_dense_adj(edge_index=data.edge_index)[0].to(device)
    X_denoised = data.x.to(device)
    A_denoised = torch_geometric.utils.to_dense_adj(edge_index=data.edge_index)[0].to(device)
    non_binary = args.denoise_non_binary

    # Main loop for denoising
    for iteration in tqdm(range(args.denoise_iterations), desc="Denoising"):
        # Decomposition
        VX, s, U = torch.linalg.svd(X_denoised)
        la, VA = torch.linalg.eigh(A_denoised)
        # Sort the eigenvalues and eigenvectors if needed
        if args.abs_ordering:
            if iteration == 0:
                sort_idx = torch.argsort(torch.abs(la))
                resort_idx = torch.argsort(la[sort_idx])
            VA = VA[:, sort_idx]
            la_abs_sort = la[sort_idx]
        else:
            la_abs_sort = la
        N = A_denoised.shape[0]

        # Denoise the node features X first
        VX_new = copy.deepcopy(VX).to(device)
        for i in range(args.rewired_index_X):
            vx = copy.deepcopy(VX[:, i]).to(device)
            va = VA[:, N - args.rewired_index_X-offset: N-offset]
            overlap = torch.matmul(vx, va)
            maxoverlap_index = torch.argmax(torch.abs(overlap))-offset
            maxoverlap = overlap[maxoverlap_index]
            maxoverlap_index = N - args.rewired_index_X + maxoverlap_index
            VX_new[:, i] = (vx * (1 - args.rewired_ratio_X) + VA[:, maxoverlap_index]
                            * args.rewired_ratio_X * torch.sign(maxoverlap))

        SI = torch.zeros(VX.shape[0], U.shape[0]).to(device)
        SI[range(min(U.shape[0], VX.shape[0])), range(min(U.shape[0], VX.shape[0]))] = s
        if args.use_right_eigvec:
            new_X = VX_new @ SI @ U
        else:
            new_X = VX_new @ SI

        # Sparsify the node features X by thresholding or top-k if needed
        if args.use_node_attr:
            if iteration == args.denoise_iterations - 1:
                if args.denoise_X_k > 0:
                    flip_X = get_top_k_features(new_X, k=args.denoise_X_k)
                else:
                    flip_X = get_clipped_features(new_X, eps=args.denoise_X_eps)
            else:
                flip_X = new_X

        # Otherwise, flip a certain amount of the node features X if needed or just keep the new features
        else:
            if iteration == args.denoise_iterations - 1:
                if non_binary:
                    flip_X = (1 - args.rewired_ratio_X_non_binary) * X + args.rewired_ratio_X_non_binary * new_X
                elif X[X == 1].sum() == X[X > 0].sum():
                    non_binary = False
                    flip_X = copy.deepcopy(X).to(device)
                    if args.flip_number_X_1 > 0:
                        mask1 = (X == 1).to(device)
                        topk = find_largest((X - new_X) * mask1, args.flip_number_X_1)
                        flip_X[mask1 & ((X - new_X) >= topk)] = 0
                    if args.flip_number_X_0 > 0:
                        mask0 = (X == 0).to(device)
                        topk = find_largest((new_X - X) * mask0, args.flip_number_X_0)
                        flip_X[mask0 & ((new_X - X) >= topk)] = 1
                else:
                    non_binary = True
                    print(f"Using non-binary denoising for features with rate {args.rewired_ratio_X_non_binary}")
                    flip_X = (1-args.rewired_ratio_X_non_binary) * X + args.rewired_ratio_X_non_binary * new_X
            else:
                flip_X = new_X

        # Use exp(X@X.T) as the feature matrix for denoising A
        if args.kernel_X:
            from sklearn import metrics
            X_kernel = metrics.pairwise.pairwise_kernels(X_denoised.cpu(), Y=None, metric='rbf')
            X_kernel = torch.tensor(X_kernel).to(device)
            _, VX = torch.linalg.eigh(X_kernel)
        else:
            VX, s, U = torch.linalg.svd(X_denoised)

        # Denoise the adjacency matrix A
        VA_new = copy.deepcopy(VA).to(device)
        for i in range(N - args.rewired_index_A-offset, N-offset):
            va = copy.deepcopy(VA[:, i]).to(device)
            vx = VX[:, :args.rewired_index_A]
            overlap = torch.matmul(va, vx)
            maxoverlap_index = torch.argmax(torch.abs(overlap))
            maxoverlap = overlap[maxoverlap_index]
            VA_new[:, i] = (va * (1 - args.rewired_ratio_A) + VX[:, maxoverlap_index]
                            * args.rewired_ratio_A * torch.sign(maxoverlap))
        # Order the eigenvalues if needed
        if args.abs_ordering:
            la_abs_sort = la_abs_sort[resort_idx]
            VA_new = VA_new[:, resort_idx]
        else:
            pass
        new_A = VA_new @ torch.diag(la_abs_sort) @ VA_new.T

        # Sparsify by threshold or top-k the adjacency matrix A if needed and build a weighted A
        if args.use_edge_attr:
            if iteration == args.denoise_iterations - 1:
                if args.denoise_A_k > 0:
                    flip_A = get_top_k_matrix(new_A, k=args.denoise_A_k)
                elif args.denoise_A_eps > 0:
                    flip_A = get_clipped_matrix(new_A, eps=args.denoise_A_eps)
            else:
                flip_A = new_A

        # Otherwise filp a certain amount of A to stay binary
        else:
            if iteration == args.denoise_iterations - 1:
                flip_A = copy.deepcopy(A).to(device)
                if args.flip_number_A_1 > 0:
                    mask1 = (A == 1).to(device)
                    topk = find_largest((A - new_A) * mask1, args.flip_number_A_1)
                    flip_A[mask1 & ((A - new_A) >= topk)] = 0

                if args.flip_number_A_0 > 0:
                    mask0 = (A == 0).to(device)
                    topk = find_largest((new_A - A) * mask0, args.flip_number_A_0)
                    flip_A[mask0 & ((new_A - A) >= topk)] = 1
            else:
                flip_A = new_A

        # Update the node features and adjacency matrix if flags are True
        if args.denoise_A:
            A_denoised = copy.deepcopy(flip_A)
        if args.denoise_x:
            X_denoised = copy.deepcopy(flip_X)

    # Otherwise just keep the original data
    if not args.denoise_A:
        A_denoised = copy.deepcopy(A)
    if not args.denoise_x:
        X_denoised = copy.deepcopy(X)

    # Create a new data object with the denoised features and adjacency matrix
    if args.use_edge_attr:
        new_data = Data(x=X_denoised, edge_index=torch_geometric.utils.dense_to_sparse(A_denoised)[0],
                        edge_attr=torch_geometric.utils.dense_to_sparse(A_denoised)[1], y=data.y).to(device)
    else:
        new_data = Data(x=X_denoised, edge_index=torch_geometric.utils.dense_to_sparse(A_denoised)[0],
                        y=data.y).to(device)

    # Log or print the results
    if args.wandb_log:
        import wandb
        if args.use_edge_attr:
            A_denoised = torch_geometric.utils.to_dense_adj(edge_index=new_data.edge_index).squeeze()
        if args.use_node_attr:
            wandb.run.summary["x_value/original"] = abs(X).sum().item()
            wandb.run.summary["x_value/denoised"] = abs(X_denoised).sum().item()
            wandb.run.summary["x_value/change"] = abs(X - X_denoised).sum().item()
        wandb.run.summary["edges/original"] = (A == 1).sum().item()
        wandb.run.summary["edges/denoised"] = (A_denoised == 1).sum().item()
        wandb.run.summary["edges/add"] = ((A-A_denoised) == -1).sum().item()
        wandb.run.summary["edges/remove"] = ((A-A_denoised) == 1).sum().item()
        wandb.run.summary["x_entries/original"] = (X == 1).sum().item()
        wandb.run.summary["x_entries/denoised"] = (X_denoised == 1).sum().item()
        wandb.run.summary["x_entries/add"] = ((X-X_denoised) == -1).sum().item()
        wandb.run.summary["x_entries/remove"] = ((X-X_denoised) == 1).sum().item()
        if non_binary:
            wandb.run.summary["x_value/original"] = abs(X).sum().item()
            wandb.run.summary["x_value/denoised"] = abs(X_denoised).sum().item()
            wandb.run.summary["x_value/change"] = abs(X - X_denoised).sum().item()
        wandb.run.summary["align"] = compute_alignment(X, A, args)
        wandb.run.summary["align_denoised"] = compute_alignment(X_denoised, A_denoised, args)

    else:
        if args.use_edge_attr:
            A_denoised = torch_geometric.utils.to_dense_adj(edge_index=new_data.edge_index).squeeze()
        if args.use_node_attr:
            print("Value of x_entries in original X: ", abs(X).sum().item())
            print("Value of x_entries in denoised X: ", abs(X_denoised).sum().item())
            print("Value of x_entries change: ", abs(X - X_denoised).sum().item())
        print("Number of edges in original A: ", (A == 1).sum().item())
        print("Number of edges denoised A: ", (A_denoised == 1).sum().item())
        print("Number of edges_added: ", ((A - A_denoised) == -1).sum().item())
        print("Number of edges_removed: ", ((A - A_denoised) == 1).sum().item())
        print("Number of x_entries in original X: ", (X == 1).sum().item())
        print("Number of x_entries in denoised X: ", (X_denoised == 1).sum().item())
        print("Number of x_entries_added: ", ((X - X_denoised) == -1).sum().item())
        print("Number of x_entries_removed: ", ((X - X_denoised) == 1).sum().item())
        if non_binary:
            print("Value of x_entries in original X: ", abs(X).sum().item())
            print("Value of x_entries in denoised X: ", abs(X_denoised).sum().item())
            print("Value of x_entries change: ", abs(X - X_denoised).sum().item())
        print("Alignment of X and A: ", compute_alignment(X, A, args))
        print("Alignment of X_denoised and A_denoised: ", compute_alignment(X_denoised, A_denoised, args))
    return new_data


def denoise_jointly_large(data, args, device):
    """
    Denoises a large graph jointly using the given arguments and truncated SVD/eigsh
    Args:
        data: PyG data object
        args: arguments from argparse
        device: device to run the operations on (cuda or cpu)

    Returns: denoised PyG data object
    """
    import scipy.sparse.linalg as sp_linalg
    offset = args.denoise_offset
    X = data.x
    A = torch_geometric.utils.to_dense_adj(edge_index=data.edge_index)[0]
    non_binary = args.denoise_non_binary

    # Main loop for denoising
    for iteration in tqdm(range(args.denoise_iterations), desc="Denoising"):
        # Decomposition
        VX, s, U = sp_linalg.svds(X.numpy(), k=args.rewired_index_X)
        VX = torch.tensor(VX.copy())
        s = torch.tensor(s.copy())
        U = torch.tensor(U.copy())
        sort_idx_X = torch.argsort(s, descending=True)
        VX = VX[:, sort_idx_X]
        s = s[sort_idx_X]
        U = U[sort_idx_X, :]
        if args.abs_ordering:
            la, VA = sp_linalg.eigsh(A.numpy(), k=(args.rewired_index_A+offset))
            la = torch.tensor(la.copy())
            VA = torch.tensor(VA.copy())
        else:
            la, VA = sp_linalg.eigsh(A.numpy(), k=2*(args.rewired_index_A+offset))
            la = torch.tensor(la.copy())
            VA = torch.tensor(VA.copy())
            sort_idx_A = torch.argsort(la, descending=True)
            la = la[sort_idx_A]
            VA = VA[:, sort_idx_A]
            la = la[:args.rewired_index_A]
            VA = VA[:, :len(la)]
        N = A.shape[0]

        # Denoise the node features X first
        VX_new = copy.deepcopy(VX).to(device)
        for i in range(args.rewired_index_X):
            vx = copy.deepcopy(VX[:, i]).to(device)
            va = VA[:, offset:args.rewired_index_A+offset].to(device)
            overlap = torch.matmul(vx, va)
            maxoverlap_index = torch.argmax(torch.abs(overlap))-offset
            maxoverlap = overlap[maxoverlap_index]
            VX_new[:, i] = (vx * (1 - args.rewired_ratio_X) + VA[:, maxoverlap_index].to(device)
                            * args.rewired_ratio_X * torch.sign(maxoverlap))

        SI = torch.zeros(args.rewired_index_X, args.rewired_index_X)
        SI[range(min(U.shape[0], VX.shape[0])), range(min(U.shape[0], VX.shape[0]))] = s
        new_X = X - VX @ SI @ U + VX_new.cpu() @ SI @ U

        # Sparsify the node features X by thresholding or top-k if needed
        if args.use_node_attr:
            if iteration == args.denoise_iterations - 1:
                if args.denoise_X_k > 0:
                    flip_X = get_top_k_features(new_X, k=args.denoise_X_k)
                else:
                    flip_X = get_clipped_features(new_X, eps=args.denoise_X_eps)
            else:
                flip_X = new_X

        # Otherwise, flip a certain amount of the node features X if needed or just keep the new features
        else:
            if iteration == args.denoise_iterations - 1:
                if non_binary:
                    flip_X = (1 - args.rewired_ratio_X_non_binary) * X + args.rewired_ratio_X_non_binary * new_X
                elif X[X == 1].sum() == X[X > 0].sum():
                    non_binary = False
                    flip_X = copy.deepcopy(X).to(device)
                    if args.flip_number_X_1 > 0:
                        mask1 = (X == 1).to(device)
                        topk = find_largest((X - new_X) * mask1, args.flip_number_X_1)
                        flip_X[mask1 & ((X - new_X) >= topk)] = 0
                    if args.flip_number_X_0 > 0:
                        mask0 = (X == 0).to(device)
                        topk = find_largest((new_X - X) * mask0, args.flip_number_X_0)
                        flip_X[mask0 & ((new_X - X) >= topk)] = 1
                else:
                    non_binary = True
                    print(f"Using non-binary denoising for features with rate {args.rewired_ratio_X_non_binary}")
                    flip_X = (1-args.rewired_ratio_X_non_binary) * X + args.rewired_ratio_X_non_binary * new_X
            else:
                flip_X = new_X

        if args.denoise_x:
            X = copy.deepcopy(flip_X)
            del flip_X, new_X
        # Denoise the adjacency matrix A
        VA_new = copy.deepcopy(VA).to(device)
        for i in range(offset, args.rewired_index_A+offset):
            va = copy.deepcopy(VA[:, i]).to(device)
            vx = VX[:, :args.rewired_index_A].to(device)
            overlap = torch.matmul(va, vx)
            maxoverlap_index = torch.argmax(torch.abs(overlap))
            maxoverlap = overlap[maxoverlap_index]
            VA_new[:, i] = (va * (1 - args.rewired_ratio_A) + VX[:, maxoverlap_index].to(device)
                            * args.rewired_ratio_A * torch.sign(maxoverlap))
        new_A = A + VA_new.cpu() @ torch.diag(la) @ VA_new.cpu().T - VA @ torch.diag(la) @ VA.T

        # Sparsify by threshold or top-k the adjacency matrix A if needed and build a weighted A
        if args.use_edge_attr:
            if iteration == args.denoise_iterations - 1:
                if args.denoise_A_k > 0:
                    flip_A = get_top_k_matrix(new_A, k=args.denoise_A_k)
                elif args.denoise_A_eps > 0:
                    flip_A = get_clipped_matrix(new_A, eps=args.denoise_A_eps)
            else:
                flip_A = new_A

        # Otherwise filp a certain amount of A to stay binary
        else:
            if iteration == args.denoise_iterations - 1:
                flip_A = copy.deepcopy(A).to(device)
                if args.flip_number_A_1 > 0:
                    mask1 = (A == 1).to(device)
                    topk = find_largest((A - new_A) * mask1, args.flip_number_A_1)
                    flip_A[mask1 & ((A - new_A) >= topk)] = 0

                if args.flip_number_A_0 > 0:
                    mask0 = (A == 0).to(device)
                    topk = find_largest((new_A - A) * mask0, args.flip_number_A_0)
                    flip_A[mask0 & ((new_A - A) >= topk)] = 1
            else:
                flip_A = new_A

        # Update the node features and adjacency matrix if flags are True
        if args.denoise_A:
            A = copy.deepcopy(flip_A)
        del VX, s, U, la, VA, VX_new, VA_new, new_A, flip_A, va, vx

    # Otherwise just keep the original data
    if not args.denoise_A:
        A = copy.deepcopy(A)
    if not args.denoise_x:
        X = copy.deepcopy(X)

    # Create a new data object with the denoised features and adjacency matrix
    if args.use_edge_attr:
        new_data = Data(x=X, edge_index=torch_geometric.utils.dense_to_sparse(A)[0],
                        edge_attr=torch_geometric.utils.dense_to_sparse(A)[1], y=data.y, train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask).to(device)
    else:
        new_data = Data(x=X, edge_index=torch_geometric.utils.dense_to_sparse(A)[0],
                        y=data.y, train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask).to(device)

    # Log or print the results
    if args.wandb_log:
        import wandb
        wandb.run.summary["edges/denoised"] = (A != 0).sum().item()
    else:
        print("Number of edges denoised A: ", (A != 0).sum().item())
    return new_data