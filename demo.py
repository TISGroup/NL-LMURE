import skimage
import numpy as np
import torch
import time
from denoise import set_seed, denoise
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def benchmark(gt: np.ndarray, test: np.ndarray, data_range: int):
    # N, H, W or N, H, W, C
    psnr_list = []
    ssim_list = []
    for i in range(gt.shape[0]):
        gt_frame = gt[i]
        test_frame = test[i]
        if gt_frame.ndim == 2:
            channel_axis = None
        elif gt_frame.ndim == 3:
            channel_axis = 2
        else:
            raise ValueError(f'Unsupported frame shape: {gt_frame.shape}')

        psnr = peak_signal_noise_ratio(gt_frame, test_frame, data_range=data_range)
        ssim = structural_similarity(gt_frame, test_frame, data_range=data_range, channel_axis=channel_axis)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    return psnr_list, ssim_list


set_seed(42)
data_range = 255.
data_name = "hypersmooth"
clean_path = f"./data/clean/{data_name}.tif"
noise_type = "gamma"
variance = 0.01
noisy_path = f"./data/noisy/{noise_type}/v={variance}/{data_name}/noisy.tif"

print(f"data_name: {data_name}, noise_type: {noise_type}, variance: {variance}")

clean = skimage.io.imread(clean_path)  #[N, H, W, C]
noisy = skimage.io.imread(noisy_path)  #[N, H, W, C]

# bechmark noisy
noisy_psnr_list, noisy_ssim_list = benchmark(clean, noisy, data_range=data_range)
print(f'noisy Avg. PSNR: {np.mean(noisy_psnr_list):.4f}, Avg. SSIM: {np.mean(noisy_ssim_list):.4f}')

# denoise
variance_params_dict = {
0.01:(4, 7, 22, 5, 22),
0.04:(3, 7, 22, 5, 22),
0.25:(2, 9, 16, 7, 22),
1:(2, 9, 16, 7, 22),
}

noisy_torch = torch.from_numpy(noisy).float().permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
temp_depth, patch_size1, topk1, patch_size2, topk2 = variance_params_dict[variance]
cuda = False
verbose = True
block_size = 37
print(f'Hparams: temp_depth: {temp_depth}, patch_size1: {patch_size1}, topk1: {topk1}, patch_size2: {patch_size2}, topk2: {topk2}')

start_time = time.time()
denoised_torch = denoise(noisy_torch, block_size=block_size, temp_depth=temp_depth,
                            patch_size1=patch_size1, topk1=topk1, patch_size2=patch_size2, topk2=topk2,
                            variance=variance, cuda=cuda, verbose=verbose)
time_elapsed = time.time() - start_time
print(f'Denoising time: {time_elapsed:.4f} sec')

denoised = denoised_torch.permute(0, 2, 3, 1).cpu().numpy()  # [N, C, H, W] -> [N, H, W, C]
denoised_psnr_list, denoised_ssim_list = benchmark(clean, denoised, data_range=data_range)
print(f'denoised Avg. PSNR: {np.mean(denoised_psnr_list):.4f}, Avg. SSIM: {np.mean(denoised_ssim_list):.4f}')
