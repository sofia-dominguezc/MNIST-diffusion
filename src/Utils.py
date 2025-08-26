# Functions to generate and classify images
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import math

# Functions regarding flow models

def flow_fn(denoiser, x, t, y, w=1):
    """Returns the flow using the denoiser(x, t, y) NN
    Assumes everything is in the right device"""
    no_y = torch.zeros_like(y)
    return w * denoiser(x, t, y) + (1 - w) * denoiser(x, t, no_y)

def score_fn(denoiser, x, t, y, w=1):
    """Returns the score asuming that denoiser is the flow.
    Assumes everything is in the right device"""
    return (t * flow_fn(denoiser, x, t, y, w=w) - x)/(1 - t)

def EulerMaruyama(f, g, x0, num_steps=200, t0=0, t1=1):
    """
    Returns the solution of dx = f(x, t)dt + g(x, t)dW at t=1
    Assumes both x0 and output are in device
    """
    h = (t1 - t0) / num_steps
    t_shape = tuple(b if i == 0 else 1 for i, b in enumerate(x0.shape))
    t = t0 + torch.zeros(t_shape, device=x0.device)
    x = x0
    for _ in range(num_steps):
        ep = torch.normal(torch.zeros_like(x), torch.ones_like(x))
        x = x + f(x, t) * h + g(x, t) * np.sqrt(h) * ep
        t += h
    return x

def forward_flow(flow_nn, z0, y, w, sigma_fn, num_steps, t0=0, t1=1):
    """
    Runs the differential equation to get z1 from z0.
    Assumes flow_nn, z0, y are in the right device
    z0: (B, *flow_nn.z_shape)
    y: (B, num_classes)
    z1: (B, *flow_nn.z_shape)
    """
    f = lambda x, t: flow_fn(flow_nn, x, t, y, w) + \
            0.5 * sigma_fn(t)**2 * score_fn(flow_nn, x, t, y, w)
    g = lambda x, t: sigma_fn(t)
    z1 = EulerMaruyama(f, g, z0, num_steps, t0=t0, t1=t1)
    return z1

def backward_flow(flow_nn, z0, y, w, sigma_fn, num_steps):
    """
    Runs the differential equation to get z1 from z0.
    Assumes flow_nn, z0, y are in the right device
    z0: (B, *flow_nn.z_shape)
    y: (B, num_classes)
    z1: (B, *flow_nn.z_shape)
    """
    opp_t = lambda t: 1 - 1/num_steps - t  # beware of division by 0
    f = lambda x, t: - flow_fn(flow_nn, x, opp_t(t), y, w) - \
            0.5 * sigma_fn(opp_t(t))**2 * score_fn(flow_nn, x, opp_t(t), y, w)
    g = lambda x, t: sigma_fn(opp_t(t))
    z1 = EulerMaruyama(f, g, z0, num_steps)
    return z1

def flow_classify(flow_nn, x, y, num_steps=100):
    """
    Finds the relative probability density of each image
    conditioned on each of the labels in y
    Assumes flow_nn, x, y are in the right device
    x: (B, *flow_nn.z_shape)
    y: (B, r), contains the idx of labels at each of B points
    out: (B, r)
    """
    # pre-process data (un-flattened)
    (B, r), z_shape = y.shape, flow_nn.z_shape
    z0 = x.repeat(r, 1, *(1 for _ in z_shape))  # (r, B, *z_shape)
    y = torch.transpose(y, 0, 1)  # (r, B)
    # run integration (flatten first)
    z0 = z0.view(B*r, *z_shape)  # (B*r, *z_shape)
    y = F.one_hot(
        y.to(torch.long).flatten(),
        num_classes=flow_nn.num_classes
    ).to(torch.float32)  # (B*r, num_classes)
    z1 = backward_flow(
        flow_nn, z0, y, w=1, sigma_fn=lambda t: 0, num_steps=num_steps,
    )
    # get probability densities (un-flatten again)
    z1 = z1.unflatten(0, (r, B)).flatten(2)  # (r, B, -1)
    probs = torch.exp(- torch.sum(z1**2, dim=-1) / 2)  # (r, B)
    return probs.transpose(0, 1)  # (B, r)

# Functions for visualization

def plot_images(images, figsize=(12, 4)):
    """images: hxw array of 2d images to plot"""
    h, w = len(images), len(images[0])
    fig, axs = plt.subplots(h, w, figsize=figsize)
    if h == 1: axs = [axs]  # edge case
    for i in range(h):
        for j in range(w):
            axs[i][j].imshow(images[i][j], cmap='gray')
            axs[i][j].axis('off')
    plt.tight_layout()
    plt.show()

def plot_interpolation(autoencoder, imgs, s=4, scale=1.4, device='cuda'):
    """
    vae: bool or float with the randomness in encoding. 0.01 is standard
    imgs: (4, 1, 28, 28) tensor of images for the corners
    s: grid will be size 2*s+1
    scale: how big each image should be
    """
    # TODO: sample imgs instead of requiring it
    h, w = 2*s + 1, 2*s + 1
    # get embeddings of the images
    with torch.no_grad():
        autoencoder.to(device)
        z_c = autoencoder.encode(imgs.to(device))

    def extract_coeffs(pos):
        """pos: (i, j) in the grid
        Returns (a, b, c) s.t. a*c1 + b*c2 + c*c3 = pos,
        where all coeffs sum to 1
        (ci are the corners of the grid)"""
        # centering grid: c1 -> -1, c2 -> -i, c3 -> +i
        pos = (pos[0] + pos[1]*1j - s - s*1j) / (s + s*1j)
        i, j = pos.real, pos.imag
        # a + b + c = 1
        # - a = i, c - b = j
        return torch.tensor(
            [-i, (1 - j + i)/2, (1 + j + i)/2],
            device=device
        ).view(-1, 1, 1, 1)

    reconstructions = np.zeros((h, w, 28, 28))
    for i in range(h):
        for j in range(w):
            if j <= h - 1 - i:
                coeffs = extract_coeffs((i, j))
                z = torch.sum(coeffs * z_c[:-1], dim=0, keepdim=True)
            else:
                coeffs = extract_coeffs((h - 1 - i, w - 1 - j))
                z = torch.sum(coeffs * z_c[1:], dim=0, keepdim=True)
            x_hat = autoencoder.decode(z)
            x_hat = torch.clip(x_hat[:, 0], 0, 1)
            reconstructions[i, j] = x_hat.cpu().detach()[0]

    plot_images(np.array(reconstructions), figsize=(scale*w, scale*h))

def plot_vae_generation(model, num_img=4, device='cuda'):
    """Generate random images from the VAE latent space"""
    shape, dtype = (num_img, *model.z_shape), torch.float32
    z = torch.normal(
        torch.zeros(shape, dtype=dtype, device=device),
        torch.ones(shape, dtype=dtype, device=device),
    )
    with torch.no_grad():
        x_hat = torch.clip(model.decode(z)[:, 0], 0, 1)
    plot_images([x_hat.cpu().detach()], fig_size=(1.4*num_img, 3))

# Miscelaneous

merged_EMNIST_names = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
)
merged_EMNIST_labels = {
    name: idx for idx, name in enumerate(merged_EMNIST_names)
}

unmerged_EMNIST_names = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)
unmerged_EMNIST_labels = {
    name: idx for idx, name in enumerate(unmerged_EMNIST_names)
}

def EMNIST_get_name(y, split="balanced"):
    """
    y: (B, ) array or list of labels
    split: 'balanced', 'byclass', or 'bymerge'
    returns: 1d list of str
    """
    def _name(idx, merge):
        """Maps an index in [0, 46] or [0, 61] to a str"""
        if merge:
            merged_EMNIST_names[idx]
        unmerged_EMNIST_names[idx]

    assert split in ["balanced", "byclass", "bymerge"]
    merge = not split == "byclass"

    return [_name(idx.item(), merge) for idx in y]

def EMNIST_get_label(text, split="balanced"):
    """
    text: str of characters
    split: 'balanced, 'byclass', or 'bymerge'
    returns: 1d array with labels
    TODO: make this work when text contains characters like c or i
    """
    def _label(name, merge):
        """Maps a str of length 1 to index in [0, 46] or [0, 61]"""
        if merge:
            merged_EMNIST_labels[name]
        unmerged_EMNIST_labels[name]
    
    assert split in ["balanced", "byclass", "bymerge"]
    merge = not split == "byclass"

    return [_label(name.item(), merge) for name in text]

def plot_var_vae(vae, loader, device='cuda'):
    """Graph the variances in the encoding space.
    The idea is that if many of them are near 1, then the model is not
    using most of the dimensions and is just trying to fill out space."""
    for x, y in loader:
        x = x.to(device)
        break
    vae.to(device)
    mu, var = vae.get_encoding(x)
    plt.hist(var.cpu().detach().view(-1), bins=20)
    plt.show()
