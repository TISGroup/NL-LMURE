import scipy
import os
import skimage
import numpy as np

def simulate_gamma_noise(x: np.ndarray, v: float, debug=False):
    """
    x (np.ndarray): Input data
    v (float): Variance of the noise
    """

    alpha = 1 / v
    beta = 1 / v

    k = alpha
    theta = 1 / beta
    w = np.random.gamma(shape=k, scale=theta, size=x.shape)
    if debug:
        print(f'sample.mean: {w.mean()}, std: {w.std()}, var: {w.var()}')
        print(f'dist.mean: {alpha / beta}, std: {np.sqrt(alpha/(beta**2))}, var: {alpha/(beta**2)}')

    return x * w, w

def simulate_normal_noise(x: np.ndarray, v: float, debug=False):
    """
    x (np.ndarray): Input data
    v (float): Variance of the noise
    """

    mu = 1
    sigma = v ** 0.5
    w = np.random.normal(loc=mu, scale=sigma, size=x.shape)
    if debug:
        print(f'sample.mean: {w.mean()}, std: {w.std()}')
        print(f'dist.mean: {mu}, std: {sigma}')

    return x * w, w

def simulate_lognormal_noise(x: np.ndarray, v: float, debug=False):
    """
    x (np.ndarray): Input data
    v (float): Variance of the noise
    """
    s2 = np.log(1+v)
    s = np.sqrt(s2)
    mu = - s2 / 2

    w = np.random.lognormal(mean=mu, sigma=s, size=x.shape)
    if debug:
        print(f'sample.mean: {w.mean()}, std: {w.std()}')
        print(f'dist.mean: {1}, std: {v**0.5}')
    
    return x * w, w
    
def simulate_wald_noise(x: np.ndarray, v: float, debug=False):
    """
    x (np.ndarray): Input data
    v (float): Variance of the noise
    """

    mu = 1
    lamb = 1 /v

    w = np.random.wald(mean=mu, scale=lamb, size=x.shape)
    if debug:
        print(f'sample.mean: {w.mean()}, std: {w.std()}')
        print(f'dist.mean: {1}, std: {v**0.5}')

    return x * w, w
    
def simulate_beta_prime_noise(x: np.ndarray, v: float, debug=False, seed=42):
    """
    x (np.ndarray): Input data
    v (float): Variance of the noise
    """

    a = 1 + 2 / v
    b = 2 + 2 / v

    w = scipy.stats.betaprime.rvs(a, b, size=x.shape, random_state=seed)
    if debug:
        print(f'sample.mean: {w.mean()}, std: {w.std()}')
        print(f'dist.mean: {1}, std: {v**0.5}')
    
    return x * w, w

def simulate_noise(x: np.ndarray, noise_type: str, variance: float, debug=False):
    """
    x (np.ndarray): Input data
    noise_type (str): Type of noise to simulate. Options: 'gamma', 'normal', 'lognormal', 'wald', 'beta_prime'
    variance (float): Variance of the noise
    debug (bool): If True, print debug information
    """

    if noise_type == 'gamma':
        return simulate_gamma_noise(x, variance, debug)
    elif noise_type == 'normal':
        return simulate_normal_noise(x, variance, debug)
    elif noise_type == 'lognormal':
        return simulate_lognormal_noise(x, variance, debug)
    elif noise_type == 'wald':
        return simulate_wald_noise(x, variance, debug)
    elif noise_type == 'beta_prime':
        return simulate_beta_prime_noise(x, variance, debug)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

# fix seed
seed = 42
np.random.seed(seed)

clean_dir = "./data/clean"
clean_files = [f for f in os.listdir(clean_dir) if f.endswith('.tif')]
noise_type = "gamma"
# noise_type = "normal"
# noise_type = "lognormal"
# noise_type = "wald"
# noise_type = "beta_prime"

noisy_dir = f"./data/noisy/{noise_type}"
if not os.path.exists(noisy_dir):
    os.makedirs(noisy_dir)

variance_list = [0.01, 0.04, 0.25, 1]

for variance in variance_list:
    for clean_file in clean_files:
        clean_path = os.path.join(clean_dir, clean_file)
        clean = skimage.io.imread(clean_path)  # [H, W] or [H, W, C]
        noisy, noise = simulate_noise(clean, noise_type=noise_type, variance=variance, debug=True)

        save_dir = os.path.join(noisy_dir, f'v={variance}', f'{clean_file.split(".")[0]}')

        print(f"Save to {save_dir} ...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        noisy_save_path = os.path.join(save_dir, f'noisy.tif')
        noise_save_path = os.path.join(save_dir, f'noise.tif')

        skimage.io.imsave(noisy_save_path, noisy)
        skimage.io.imsave(noise_save_path, noise)

        print(f"Saved noisy data to {noisy_save_path} and noise to {noise_save_path}")