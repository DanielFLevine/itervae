import json
import os
from dataclasses import dataclass, field, asdict
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from transformers import HfArgumentParser
from scipy.linalg import sqrtm
from torch.optim import Adam
from torchvision.datasets import MNIST, EMNIST, CIFAR10
from torchvision.models import inception_v3
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader
from pathlib import Path


from IterCVAE import IterCVAE
from train_utils import train_model
from utils.pytorch_fid_np import run_fid


@dataclass
class TrainingArguments:
    dataset: str = field(
        default="CIFAR10"
        )
    in_channels: int = field(
        default=3
        )
    hidden_channels: int = field(
        default=16
        )
    kernel_size: int = field(
        default=3
        )
    latent_dim: int = field(
        default=50
        )
    num_convs: int = field(
        default=1
        )
    num_iters: int = field(
        default=1
        )
    epochs: int = field(
        default=100
        )
    dataset_path: str = field(
        default="~/datasets"
        )
    batch_size: int = field(
        default=100
        )
    lr: float = field(
        default=1e-3
        )
    beta: float = field(
        default=5.0
        )
    gamma: float = field(
        default=1.0
        )
    device_str: str = field(
        default="auto"
        )
    

@dataclass
class FIDArguments:
    fid_num_workers: int = field(
        default=1
        )
    fid_dims: int = field(
        default=2048
        )
    fid_save_stats: bool = field(
        default=True
        )
    fid_path: str = field(
        default="fid_metrics"
        )
    fid_stats_path: str = field(
        default="fid_metrics/saved_stats"
        )
    fid_validation_path: str = field(
        default="fid_metrics/cifar10_test.npy"
        )
    fid_validation_stats_path: str = field(
        default="fid_metrics/saved_stats/cifar10_test_stats.npz"
        )
    

def get_inception_for_fid(device):

    inception = inception_v3(weights="IMAGENET1K_V1", transform_input=False).to(device).eval()

    return_nodes = {
        "avgpool":"avgpool"
    }

    return create_feature_extractor(inception, return_nodes=return_nodes)


def compute_frechet_distance(real_data, generated_data, model, device, batch_size):

    real_activations = []
    generated_activations = []
    
    for real_batch, generated_batch in zip(real_data, generated_data):
        real_batch = real_batch[0].to(device)
        generated_batch = generated_batch.to(device)
        real_batch = nn.functional.interpolate(real_batch, size=(299, 299), mode='bilinear', align_corners=False)
        generated_batch = nn.functional.interpolate(generated_batch, size=(299, 299), mode='bilinear', align_corners=False)
        real_batch = 2 * real_batch - 1  # Scale from range (0, 1) to range (-1, 1)
        generated_batch = 2 * generated_batch - 1  # Scale from range (0, 1) to range (-1, 1)


        with torch.no_grad():
            real_activations.append(model(real_batch)["avgpool"].reshape(batch_size, 2048).cpu().numpy())
            generated_activations.append(model(generated_batch)["avgpool"].reshape(batch_size, 2048).cpu().numpy())
    
    real_activations = np.concatenate(real_activations, axis=0)
    generated_activations = np.concatenate(generated_activations, axis=0)
    
    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    mu_generated = np.mean(generated_activations, axis=0)
    sigma_generated = np.cov(generated_activations, rowvar=False)
    
    diff = mu_real - mu_generated
    cov_mean = sqrtm(sigma_real.dot(sigma_generated))
    fid = np.real(diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_generated) - 2 * np.trace(cov_mean))
    
    return fid


def get_loaders(ds, dataset_path, batch_size):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    if ds == EMNIST:
        train_dataset = ds(dataset_path, split="balanced", transform=image_transform, train=True, download=True)
        test_dataset  = ds(dataset_path, split="balanced", transform=image_transform, train=False, download=True)
    else:
        train_dataset = ds(dataset_path, transform=image_transform, train=True, download=True)
        test_dataset  = ds(dataset_path, transform=image_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


def generate_fid_samples(model, batch_size, num_samples, im_shape, np_save_path=None, save=False):
    model.eval()
    generated_samples = []
    with torch.no_grad():
        for _ in tqdm(range(num_samples // batch_size)):
            noise = torch.randn(batch_size, model.latent_dim, device=device)
            generated_batch = model.decode(noise)
            generated_samples.append(generated_batch.cpu().view(im_shape))
    generated_samples = torch.cat(generated_samples, dim=0).cpu().numpy()
    if save:
        with open(np_save_path, "wb") as f:
            np.save(np_save_path, generated_samples)
    return generated_samples


def plot_loss(epochs, hidden_dim, losses, loss_name, iter_type, ds_name):

    x = [i for i in range(epochs)]
    
    for i, y in enumerate(losses):
        plt.plot(x, y, label = f"{i+1} {iter_type}")
    plt.legend()
    plt.title(f"{ds_name}\n{loss_name} Over Epochs With Hidden Dim {hidden_dim}")
    plt.xlabel('Epochs') 
    plt.ylabel('Loss')

    save_path = f"plots/{iter_type}"
    if not os.path.exists(save_path):
        Path(save_path).mkdir(parents=False, exist_ok=False)
    file_name = f"/{ds_name}_hidden_dim_{hidden_dim}_{loss_name}"
    plt.savefig(save_path+file_name)
    plt.close()


def train_pipeline(
        hidden_dim,
        latent_dim,
        dataset,
        num_convs,
        in_channels,
        hidden_channels,
        kernel_size,
        num_iters,
        dataset_path,
        batch_size,
        epochs,
        lr,
        device,
        beta,
        gamma,
        fid_args
        ):
    name_to_dataset = {"CIFAR10": CIFAR10, "MNIST": MNIST, "EMNIST": EMNIST}
    ds = name_to_dataset[dataset]
    train_loader, test_loader = get_loaders(ds, dataset_path, batch_size)
    iter_test_loader = cycle(iter(test_loader))
    model = IterCVAE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        kernel_size=kernel_size,
        num_convs=num_convs,
        num_iters=num_iters,
        device=device).to(device)
    wandb.watch(model, log="all", log_freq=100)
    
    optimizer = Adam(model.parameters(), lr=lr)
    train_model(
        model=model,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        beta=beta,
        gamma=gamma,
        train_loader=train_loader,
        test_loader=test_loader,
        iter_test_loader=iter_test_loader,
        device=device
        )
    
    # stats_filename = f"stats_convs{num_convs}_iters{num_iters}_hiddenchannels{hidden_channels}_kersize{kernel_size}_latentdim{latent_dim}_lr{lr}.npz"
    # generated_samples_filename = f"fid_samples_convs{num_convs}_iters{num_iters}_hiddenchannels{hidden_channels}_kersize{kernel_size}_latentdim{latent_dim}_lr{lr}.npy"
    # generated_samples_path = fid_args.fid_path + "/" + generated_samples_filename

    im_shape = (batch_size, 3, 32, 32)
    num_samples = 10000
    print(f"\nGENERATING {num_samples} SAMPLES FOR FID...")
    fid_values = []
    for _ in range(5):
        generated_samples = generate_fid_samples(model, batch_size, num_samples, im_shape)
        generated_data = DataLoader(dataset=generated_samples, batch_size=batch_size, shuffle=False)
        inception_model = get_inception_for_fid(device)
        fid_value = compute_frechet_distance(test_loader, generated_data, inception_model, device, batch_size)
        fid_values.append(fid_value)


    # if not os.path.exists(fid_args.fid_stats_path):
    #     Path(fid_args.fid_stats_path).mkdir(parents=False, exist_ok=False)
    # fid_value = run_fid(
    #     dims=fid_args.fid_dims,
    #     batch_size=batch_size,
    #     num_workers=fid_args.fid_num_workers,
    #     device=device,
    #     validation_path=fid_args.fid_validation_path,
    #     generated_samples_path=generated_samples_path,
    #     stats_path=fid_args.fid_stats_path,
    #     stats_filename=stats_filename,
    #     fid_validation_stats_path=fid_args.fid_validation_stats_path
    #     )

    fid_values = np.array(fid_values)
    avg_fid = np.mean(fid_values)
    std_fid = np.std(fid_values)
    print(f"\nFIDS ARE: {fid_values}")
    print(f"\nAVERAGE FID IS: {avg_fid}")
    print(f"\nSTD FOR FID IS: {std_fid}")
    
    wandb.log({"AVERAGE FID": avg_fid})
    wandb.log({"STD FID": std_fid})


if __name__ == "__main__":
    parser = HfArgumentParser([TrainingArguments, FIDArguments])
    args, fid_args = parser.parse_args_into_dataclasses()
    if args.device_str == "cuda":
        assert torch.cuda.is_available(), "Specified CUDA in arguments, but CUDA is not available."
        device = torch.device("cuda")
    else:
        print("\nAUTO DETECTING BEST DEVICE...")
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
    print(f"\nDEVICE IS {device}")
    print(f"\nTRAINING PARAMETERS")
    print(json.dumps(asdict(args), indent=2))
    wandb.init()

    train_pipeline(
        hidden_dim=args.hidden_channels*32*32,
        latent_dim=args.latent_dim,
        dataset=args.dataset,
        num_convs=args.num_convs,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        num_iters=args.num_iters,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        beta=args.beta,
        gamma=args.gamma,
        fid_args=fid_args
        )
    
# Compute loss over trajectory - pair up forward and backward outputs for L2 loss and compare inception score
# Compare iterative VAE vs non-iterative VAE and compare inception scores - does more iterations make better outputs?
# Use learnable smoothing factors
# Also look at outputs explicitly
# Initialize smoothing at 0.5
# Make it convolutional


            
        