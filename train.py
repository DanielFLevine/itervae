import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from transformers import HfArgumentParser
from torch.optim import Adam
from torchvision.datasets import MNIST, EMNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path


from IterVAE import IterVAE
from train_utils import train_model


@dataclass
class TrainingArguments:
    dataset: str = field(
        default="CIFAR10"
        )
    hidden_dim: int = field(
        default=200
        )
    num_linears: int = field(
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
    lr: int = field(
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
        dataset,
        num_linears,
        num_iters,
        dataset_path,
        batch_size,
        epochs,
        lr,
        device,
        beta,
        gamma
        ):
    name_to_dataset = {"CIFAR10": CIFAR10, "MNIST": MNIST, "EMNIST": EMNIST}
    ds = name_to_dataset[dataset]
    train_loader, test_loader = get_loaders(ds, dataset_path, batch_size)
    x_dim = torch.numel(train_loader.dataset[0][0])
    model = IterVAE(
        input_dim=x_dim,
        hidden_dim=hidden_dim,
        latent_dim=200,
        output_dim=x_dim,
        num_linears=num_linears,
        num_iters=num_iters,
        device=device).to(device)
    
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
        device=device
        )


    # fid_scores_layers = np.array(fid_scores_layers)
    # fid_scores_iters = np.array(fid_scores_iters)
    # with open(f"fid_scores/fid_scores_layers_hdim{hidden_dim}_ds{ds_names[ds]}.npy", "wb") as f:
    #     np.save(f, fid_scores_layers)
    # with open(f"fid_scores/fid_scores_iters_hdim{hidden_dim}_ds{ds_names[ds]}.npy", "wb") as f:
    #     np.save(f, fid_scores_iters)

    # plot_loss(epochs, hidden_dim, train_losses_layers, "Training Loss", "Layers", ds_names[ds])
    # plot_loss(epochs, hidden_dim, val_losses_layers, "Validation Loss", "Layers", ds_names[ds])
    # plot_loss(epochs, hidden_dim, train_losses_iters, "Training Loss", "Iterations", ds_names[ds])
    # plot_loss(epochs, hidden_dim, val_losses_iters, "Validation Loss", "Iterations", ds_names[ds])

if __name__ == "__main__":
    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
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
        hidden_dim=args.hidden_dim,
        dataset=args.dataset,
        num_linears=args.num_linears,
        num_iters=args.num_iters,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        beta=args.beta,
        gamma=args.gamma
        )
    
# Compute loss over trajectory - pair up forward and backward outputs for L2 loss and compare inception score
# Compare iterative VAE vs non-iterative VAE and compare inception scores - does more iterations make better outputs?
# Use learnable smoothing factors
# Also look at outputs explicitly
# Initialize smoothing at 0.5
# Make it convolutional


            
        