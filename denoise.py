import numpy as np
import torch
import random
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
import warnings
from einops import rearrange, repeat
from sklearn.linear_model import LinearRegression, RANSACRegressor

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def estimate_variance(y: np.ndarray or torch.Tensor, patch_size=8, seed=42) -> float:

    if isinstance(y, np.ndarray):
        y = convert(y)  # [N, C, H, W]
    elif isinstance(y, torch.Tensor):
        pass
    else:
        raise ValueError(f'Unsupported data type: {type(y)}')

    p = F.unfold(y, kernel_size=patch_size, stride=patch_size)  # [N, C*patch_size*patch_size, n_patches]
    p = p.permute(0, 2, 1).reshape(-1, p.shape[1])  # [N*n_patches, C*patch_size*patch_size]
    p = p.numpy()
    mu = p.mean(1) ** 2
    variance = p.var(1)

    reg = RANSACRegressor(random_state=seed)
    # reg.fit(mu.reshape(-1, 1), variance)
    base = LinearRegression(fit_intercept=False)  # no intercept
    reg = RANSACRegressor(estimator=base, random_state=seed)
    reg.fit(mu.reshape(-1, 1), variance)

    return reg.estimator_.coef_.item()

def convert(data: np.ndarray):
    # assert data.ndim == 3

    data = torch.from_numpy(data).float()

    if data.ndim == 2:
        # single frame grayscale image
        data = data.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    elif data.ndim == 3:
        # multi frame image data
        # RGB or multi frame grayscale
        if data.shape[2] == 3:
            # single frame RGB image, [H, W, 3]
            data = data.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        else:
            data = data.unsqueeze(1)  # [N, 1, H, W]
    elif data.ndim == 4:
        # multi frame RGB image data, [N, H, W, 3]
        data = data.permute(0, 3, 1, 2)  # [N, 3, H, W]
    else:
        raise ValueError(f'Unsupported data shape: {data.shape}')
    
    return data # [N, C, H, W]

def block_matching(input_x, k, p, w, s, variance):
    """
    Finds similar patches within a specified window around each reference patch.

    Args:
        input_x (torch.FloatTensor): Input image tensor of shape (N, C, H, W).
        k (int): Number of most similar patches to find.
        p (int): Patch size.
        w (int): Search window size.
        s (int): Stride for moving between reference patches.

    Returns:
        torch.LongTensor: Indices of shape (N, Href, Wref, k) of similar patches for each reference patch.
    """
    if w % 2 != 1:
        raise ValueError(f"Invalid input: w ({w}) must be an odd integer.")
    if (w - p + 1) ** 2 < k:
        raise ValueError(
            f"Invalid input: k ({k}) must be less than or equal to the number of overlapping patches per window, that is {(w - p + 1) ** 2}.")

    def block_matching_aux(input_x_pad, k, p, v, s, variance):
        """
        Auxiliary function to perform block matching in a padded input tensor.

        Args:
            input_x_pad (torch.FloatTensor): Padded input tensor of shape (N, C, H, W).
            k (int): Number of similar patches to find.
            p (int): Patch size.
            v (int): Half of the search window size.
            s (int): Stride for moving between reference patches.

        Returns:
            torch.LongTensor: Indices of shape (N, Href, Wref, k) of similar patches for each reference patch.
        """
        N, C, H, W = input_x_pad.size()
        assert C == 1
        Href, Wref = -((H - (2 * v + p) + 1) // -s), -((W - (
                2 * v + p) + 1) // -s)  # ceiling division, represents the number of reference patches along each axis for unfold with stride=s
        # norm_patches = F.avg_pool2d(input_x_pad ** 2, p, stride=1)
        # faster
        norm_patches = F.avg_pool2d(input_x_pad.pow(2), p, stride=1)
        norm_patches = F.unfold(norm_patches, 2 * v + 1, stride=s)
        norm_patches = rearrange(norm_patches, 'n (p1 p2) l -> 1 (n l) p1 p2', p1=2 * v + 1)
        local_windows = F.unfold(input_x_pad, 2 * v + p, stride=s) / p
        local_windows = rearrange(local_windows, 'n (p1 p2) l -> 1 (n l) p1 p2', p1=2 * v + p)
        ref_patches = rearrange(local_windows[..., v:-v, v:-v], '1 b p1 p2 -> b 1 p1 p2')
        scalar_product = F.conv2d(local_windows, ref_patches, groups=N * Href * Wref)
        distances = 1 / (variance + 1) * norm_patches - 2 * scalar_product  # (up to a constant)
        distances[:, :, v, v] = float('-inf')  # the reference patch is always taken
        distances = rearrange(distances, '1 (n h w) p1 p2 -> n h w (p1 p2)', n=N, h=Href, w=Wref)
        indices = torch.topk(distances, k, dim=-1, largest=False,
                             sorted=False).indices  # float('nan') is considered to be the highest value for topk
        return indices

    v = w // 2
    input_x_pad = F.pad(input_x, [v] * 4, mode='constant', value=float('nan'))
    N, C, H, W = input_x.size()
    Href, Wref = -((H - p + 1) // -s), -(
            (W - p + 1) // -s)  # ceiling division, represents the number of reference patches along each axis for unfold with stride=s
    ind_H_ref = torch.arange(0, H - p + 1, step=s, device=input_x.device)
    ind_W_ref = torch.arange(0, W - p + 1, step=s, device=input_x.device)
    if (H - p + 1) % s != 1:
        ind_H_ref = torch.cat((ind_H_ref, torch.tensor([H - p], device=input_x.device)), dim=0)
    if (W - p + 1) % s != 1:
        ind_W_ref = torch.cat((ind_W_ref, torch.tensor([W - p], device=input_x.device)), dim=0)

    indices = torch.empty(N, ind_H_ref.size(0), ind_W_ref.size(0), k, dtype=ind_H_ref.dtype, device=ind_H_ref.device)
    indices[:, :Href, :Wref, :] = block_matching_aux(input_x_pad, k, p, v, s, variance)
    if (H - p + 1) % s != 1:
        indices[:, Href:, :Wref, :] = block_matching_aux(input_x_pad[:, :, -(2 * v + p):, :], k, p, v, s, variance)
    if (W - p + 1) % s != 1:
        indices[:, :Href, Wref:, :] = block_matching_aux(input_x_pad[:, :, :, -(2 * v + p):], k, p, v, s, variance)
        if (H - p + 1) % s != 1:
            indices[:, Href:, Wref:, :] = block_matching_aux(input_x_pad[:, :, -(2 * v + p):, -(2 * v + p):], k, p, v, s, variance)

    # (ind_row, ind_col) is a 2d-representation of indices
    ind_row = torch.div(indices, 2 * v + 1, rounding_mode='floor') - v
    ind_col = torch.fmod(indices, 2 * v + 1) - v

    # from 2d to 1d representation of indices
    indices = (ind_row + rearrange(ind_H_ref, 'h -> 1 h 1 1')) * (W - p + 1) + (ind_col + rearrange(ind_W_ref, 'w -> 1 1 w 1'))
    return indices

def gather_groups(input_y, indices, p):
    """
    Gathers groups of patches based on the indices from block-matching.

    Args:
        input_y (torch.FloatTensor): Input image tensor of shape (N, C, H, W).
        indices (torch.LongTensor): Indices of similar patches of shape (N, Href, Wref, k).
        k (int): Number of similar patches.
        p (int): Patch size.

    Returns:
        torch.FloatTensor: Grouped patches of shape (N, Href, Wref, k, p**2).
    """
    unfold_Y = F.unfold(input_y, p)
    _, n, _ = unfold_Y.shape
    _, Href, Wref, k = indices.shape
    Y = torch.gather(unfold_Y, dim=2, index=repeat(indices, 'N h w k -> N n (h w k)', n=n))
    return rearrange(Y, 'N n (h w k) -> N h w k n', k=k, h=Href, w=Wref)

def aggregate(X_hat, weights, indices, H, W, p):
    """
    Aggregates groups of patches back into the image grid.

    Args:
        X_hat (torch.FloatTensor): Grouped denoised patches of shape (N, Href, Wref, k, p**2).
        weights (torch.FloatTensor): Weights of each patch of shape (N, Href, Wref, k, 1).
        indices (torch.LongTensor): Indices of the patches in the original image of shape (N, Href, Wref, k).
        H (int): Height of the original image.
        W (int): Width of the original image.
        p (int): Patch size.

    Returns:
        torch.FloatTensor: Reconstructed image tensor.
    """
    N, _, _, _, n = X_hat.size()
    X = rearrange(X_hat * weights, 'N h w k n -> (N h w k) n')
    weights = repeat(weights, 'N h w k 1 -> (N h w k) n', n=n)
    offset = (H - p + 1) * (W - p + 1) * torch.arange(N, device=X.device).view(-1, 1, 1, 1)
    indices = rearrange(indices + offset, 'N h w k -> (N h w k)')

    X_sum = torch.zeros(N * (H - p + 1) * (W - p + 1), n, dtype=X.dtype, device=X.device)
    weights_sum = torch.zeros_like(X_sum)

    X_sum.index_add_(0, indices, X)
    weights_sum.index_add_(0, indices, weights)
    X_sum = rearrange(X_sum, '(N hw) n -> N n hw', N=N)
    weights_sum = rearrange(weights_sum, '(N hw) n -> N n hw', N=N)

    return F.fold(X_sum, (H, W), p) / F.fold(weights_sum, (H, W), p)

def solve_step1(Z: torch.Tensor, variance, cuda: bool) -> tuple[torch.Tensor, torch.Tensor]:
    N, Href, Wref, k, n = Z.shape

    G = torch.matmul(Z, Z.transpose(-1, -2)) # [..., k, k]

    S = Z.pow(2).sum(4)
    D = torch.diag_embed(S)

    R = G - variance / (variance + 1.0) * D

    if cuda:
        theta = torch.linalg.lstsq(G.cuda(), R.cuda()).solution.detach().cpu() # [k, k]
    else:
        theta = torch.linalg.lstsq(G, R).solution # [..., k, k]

    theta = theta.transpose(-1, -2)

    Z_hat = torch.matmul(theta, Z) # [N, H, W, n, k]

    weights = 1 / torch.sum(theta ** 2, dim=-1, keepdim=True).clip(1 / k, 1)
    
    return Z_hat, weights

def solve_step2(Y: torch.Tensor, X_hat: torch.Tensor, cuda: bool) -> tuple[torch.Tensor, torch.Tensor]:

    N, Href, Wref, k, n = Y.shape

    Y_perm = Y.permute(0, 1, 2, 4, 3)  # [N, Href, Wref, n, k]
    X_hat_perm = X_hat.permute(0, 1, 2, 4, 3)  # [N, Href, Wref, n, k]


    if cuda:
        theta = torch.linalg.lstsq(Y_perm.cuda(), X_hat_perm.cuda()).solution.detach().cpu() # [k, k]
    else:
        theta = torch.linalg.lstsq(Y_perm, X_hat_perm).solution # [..., k, k]


    theta = theta.transpose(-1, -2)

    Z_hat = torch.matmul(theta, Y) # [N, H, W, n, k]
    weights = 1 / torch.sum(theta ** 2, dim=-1, keepdim=True).clip(1 / k, 1)

    return Z_hat, weights

def fake_grayscale(input_tensor, variance):
    num_channels = input_tensor.size(1)
    _min = input_tensor.amin(dim=(1, 2, 3), keepdim=True)
    grayscale_tensor = input_tensor - _min
    grayscale_tensor = torch.log(grayscale_tensor + 1.0)
    grayscale_tensor = torch.mean(grayscale_tensor, dim=1, keepdim=True)
    grayscale_tensor = torch.exp(grayscale_tensor) - 1.0 + _min

    effective_variance = variance / num_channels
    
    return grayscale_tensor, effective_variance

def __denoise_step1(input_tensor: torch.Tensor, k: int, p: int, w: int, s: int, variance: float, cuda: bool) -> torch.Tensor:
    _, C, H, W = input_tensor.size()
    if C != 1:
        # grayscale_tensor = torch.mean(input_tensor, dim=1, keepdim=True)
        grayscale_tensor, effective_variance = fake_grayscale(input_tensor, variance)
    else:
        grayscale_tensor = input_tensor
        effective_variance = variance

    indices = block_matching(grayscale_tensor, k, p, w, s, effective_variance)
    Z = gather_groups(input_tensor, indices, p)
    Z_hat, weights = solve_step1(Z, variance, cuda)
    z_hat = aggregate(Z_hat, weights, indices, H, W, p)
    return z_hat

def __denoise_step2(input_tensor: torch.Tensor, ref_tensor: torch.Tensor, k: int, p: int, w: int, s: int, ref_variance: float, cuda: bool) -> torch.Tensor:
    _, C, H, W = input_tensor.size()
    if C != 1:
        # grayscale_ref_tensor = torch.mean(ref_tensor, dim=1, keepdim=True)
        grayscale_ref_tensor, effective_ref_variance = fake_grayscale(ref_tensor, ref_variance)
    else:
        grayscale_ref_tensor = ref_tensor
        effective_ref_variance = ref_variance

    indices = block_matching(grayscale_ref_tensor, k, p, w, s, effective_ref_variance)
    Y = gather_groups(input_tensor, indices, p)
    X_hat = gather_groups(ref_tensor, indices, p)
    Z_hat, weights = solve_step2(Y, X_hat, cuda)
    z_hat = aggregate(Z_hat, weights, indices, H, W, p)
    return z_hat

def denoise(input_tensor: Tensor, block_size: int, temp_depth: int, 
            patch_size1: int, topk1: int, patch_size2: Optional[int], topk2: Optional[int], variance:Optional[float], 
            cuda: bool = False, verbose: bool = False) -> Tensor:
    assert input_tensor.ndim == 4, f"Input tensor must be 4D, but got {input_tensor.ndim}D."
    if variance is not None:
        assert variance > 0, f"Variance must be positive, but got {variance}."
    
    n, c, h, w = input_tensor.shape

    if n == 1 and c == 1:
        if temp_depth != 1:
            warnings.warn("For single frame grayscale image, temp_depth must be 1. Setting it to 1.")
            temp_depth = 1
    elif n == 1 and c == 3:
        if temp_depth != 1:
            # raise ValueError("For single frame RGB image, block_depth and patch_depth must be 3.")
            warnings.warn("For single frame RGB image, temp_depth must be 1. Setting it to 1.")
            temp_depth = 1

    # check block_size
    _t = torch.randn(1, 1, block_size, block_size)
    _t_unfold = F.unfold(_t, kernel_size=patch_size1, stride=patch_size1)

    if _t_unfold.size(-1) < topk1:
        warnings.warn("topk1 (%d) is larger than the number of patches in a block (%d). Setting topk to %d." % (topk1, _t_unfold.size(-1) - 1, _t_unfold.size(-1) - 1))
        topk1 = _t_unfold.size(-1) - 1

    if patch_size2 is not None and topk2 is not None:
        _t_unfold = F.unfold(_t, kernel_size=patch_size2, stride=patch_size2)
        if _t_unfold.size(-1) < topk2:
            warnings.warn("topk2 (%d) is larger than the number of patches in a block (%d). Setting topk to %d." % (topk2, _t_unfold.size(-1) - 1, _t_unfold.size(-1) - 1))
            topk2 = _t_unfold.size(-1) - 1
    
    if variance is None:
        # estimate
        data_var = estimate_variance(input_tensor)
        if data_var >= 0.5:
            data_var = estimate_variance(input_tensor, patch_size=32)
            if verbose:
                print(f'Re-estimated variance with larger patch size 32: {data_var}')
        
        data_var = max(data_var, 1e-8)

        if verbose:
            print(f'Estimated variance: {data_var} for input')
    else:
        data_var = variance


    a2 = input_tensor.pow(2).sum(dim=[1, 2, 3]) # (n)
    b2 = input_tensor.pow(2).sum(dim=[1, 2, 3]) # (n)
    ab = input_tensor.reshape(n, -1) @ input_tensor.reshape(n, -1).t() # (n, n)
    dist_mat = (a2.view(-1, 1) + b2.view(1, -1)) / (variance + 1) - 2 * ab


    denoised_data = torch.zeros_like(input_tensor)

    denoised_frame_indices = set([])
    for i in range(n):

        if i in denoised_frame_indices:
            continue

        if verbose:
            print(f'Stage 1, Processing frames [{len(denoised_frame_indices)}/{n}]', end='\r' if i != n else '\n')

        row = dist_mat[i]
        frame_indices = torch.argsort(row)[:temp_depth] # include self
        # if verbose:
        #     print(f"Frame {i}, similar frames: {frame_indices}")
        data = input_tensor[frame_indices]  # [D, C, H, W]
        denoised = __denoise_step1(data, k=topk1, p=patch_size1, w=block_size, s=patch_size1, variance=data_var, cuda=cuda)
        denoised_data[frame_indices] = denoised
        denoised_frame_indices.update(frame_indices.cpu().tolist())

    if patch_size2 is not None and topk2 is not None:
        data_var_denoised = estimate_variance(denoised_data)
        if data_var_denoised >= 0.5:
            data_var_denoised = estimate_variance(denoised, patch_size=32)
            if verbose:
                print(f'Re-estimated variance with larger patch size 32: {data_var_denoised}')
        data_var_denoised = max(data_var_denoised, 1e-8)
        if verbose:
            print(f'Estimated variance: {data_var_denoised} for denoised')

        a2 = denoised_data.pow(2).sum(dim=[1, 2, 3]) # (n)
        b2 = denoised_data.pow(2).sum(dim=[1, 2, 3]) # (n)
        ab = denoised_data.reshape(n, -1) @ denoised_data.reshape(n, -1).t() # (n, n)
        dist_mat = (a2.view(-1, 1) + b2.view(1, -1)) / (variance + 1) - 2 * ab

        denoised_data2 = torch.zeros_like(input_tensor)

        denoised_frame_indices2 = set([])

        for i in range(n):
            if i in denoised_frame_indices2:
                continue

            if verbose:
                print(f'Step 2, Processing frames [{len(denoised_frame_indices2)}/{n}]', end='\r' if i != n else '\n')

            row = dist_mat[i]
            frame_indices = torch.argsort(row)[:temp_depth] # include self
            # print(f"Step 2, Frame {i}, similar frames: {frame_indices}")
            data = input_tensor[frame_indices]  # [D, C, H, W]
            denoised = denoised_data[frame_indices]  # [D, C, H, W]

            denoised = __denoise_step2(data, denoised, k=topk2, p=patch_size2, w=block_size, s=patch_size2, ref_variance=data_var_denoised, cuda=cuda)

            denoised_data2[frame_indices] = denoised
            denoised_frame_indices2.update(frame_indices.cpu().tolist())
        
        return denoised_data2
    else:
        return denoised_data
 