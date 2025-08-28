# MNIST/EMNIST/FashionMNIST

Diffusion model implementation in pytorch.

I built this as a personal project to practice using pytorch-lightning and training variational autoencoders (VAEs) and flow/diffusion models.

<div align="center">
  <img src="https://github.com/user-attachments/assets/f39e7db0-3617-4bc0-b5f9-3f718a843595" width="1600" height="960" alt="example_output"/>
</div>

To simplify the training process, the flow/diffusion model is not trained in the images directly but instead on the latent space of an autoencoder/VAE.

To generate images, first sample random noise in the latent space, then apply the flow/diffusion model, and then use the decoder to get the image.

<div align="center">
  <img src="https://github.com/user-attachments/assets/78042bd6-ec5f-443b-8831-76f3f92e7056"  width="813" height="312" alt="flow_chart"/>
</div>

To generate syntetic images using the pre-trained models, run

```python src generate --dataset EMNIST --model autoencoder```

The repository also supports training the models:
- train autoencoder/VAE: `python src train --dataset MNIST --model autoencoder` or `--model vae`
- generate dataset of encodings of the autoencoder: `python src encode-dataset --dataset MNIST --model vae`
- train diffusion model in latent space: `python src train --dataset MNIST --model flow`

The repository also has a `test` mode. Besides calculating the loss function in the test data, if the architecture is autoencoder or vae, it also calculates the prediction accuracy that we obtain using Bayes rule:

$$p(y | x) \propto p(x | y) p(y)$$

The repository also has a `test-reconstruction` mode that plots multiple images along with their reconstructions by the autoencoder/VAE, useful for evaluating its performance.

## Command Line Interface

The project can be run from the terminal with different modes. Each mode has its own arguments, but many arguments are shared.

### Common Arguments

These flags work in most modes:

- `--dataset {MNIST, EMNIST, FashionMNIST}` - which dataset to use.

- `--model {autoencoder, vae, flow}` - model architecture.

- `--model-version {dev, main}` - which checkpoint to use:
    - None: don't load parameters. default in `mode=train`.
    - dev: scratch / development checkpoint. It's overwritten each training epoch.
    - main: main checkpoint. User decides if overwriting it at the end of training. Default in all other modes.

- `--root PATH (default: data)` - root directory for datasets and parameters.

- `--batch-size (default: 128)` - batch size for DataLoaders.

- `--split {balanced, byclass, bymerge} (default: balanced)` - EMNIST split. Ignored for MNIST / FashionMNIST.

### Training (train)

Trains a model.

- All common arguments

- `--lr (default: 1e-3)` - learning rate.

- `--total-epochs (default: 10)`

- `--num-workers (default: 0)` - dataloader workers.

- `--milestones (default: [])` - epochs where to decrease learning rate.

- `--gamma (default: 0.2)` - learning rate decay factor at each milestone.

### Testing (test)
Tests a trained model.

- All common arguments.

- `--num-workers (default: 0)`

### Encode Dataset (encode-dataset)
Encodes a dataset using a trained autoencoder/VAE.

- All common arguments.

### Generation (generate)
Generates samples using a diffusion model and an autoencoder/VAE.

- All common arguments.

- `--height (default: 8)` - grid height.

- `--width (default: 8)` - grid width.

- `--scale (default: 0.8)` - scaling factor for plotting.

- `--weight (default: 3)` - classifier-free guidance weigth.

- `--diffusion (default: 0.5)` - noise level in diffusion. Corresponds to $sigma(t) = \text{diffusion} \cdot (1 - t)$.

- `--autoencoder-version {dev, main} (default: main)` - which autoencoder checkpoint to use.

### Reconstruction (test-reconstruction)
Reconstructs images from a dataset using an autoencoder/VAE.

- All common arguments.

- `--height (default: 8)`

- `--width (default: 8)`

- `--scale (default: 0.8)`

### Extra arguments
Any unknown arguments passed to the CLI are forwarded to the model constructor. For example:

```python src train --dataset MNIST --model autoencoder --dim1 64 --n_layers 3```

will instantiate the autoencoder with `dim1=64` and `n_layers=3`.
