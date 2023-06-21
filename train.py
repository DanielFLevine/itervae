import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torchvision.datasets import MNIST, EMNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path


from IterVAE import IterVAE
from train_utils import train_model



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
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, **kwargs)
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


def train_pipeline(hidden_dims, datasets, num_enc, dataset_path, batch_size, epochs, lr, device):
    ds_names = {MNIST: "MNIST", EMNIST: "EMNIST", CIFAR10: "CIFAR10"}
    for hidden_dim in hidden_dims:
        for ds in datasets:
            train_loader, test_loader = get_loaders(ds, dataset_path, batch_size)
            x_dim = torch.numel(train_loader.dataset[0][0])
            train_losses_layers = []
            val_losses_layers = []
            train_losses_iters = []
            val_losses_iters = []
            for i in range(num_enc):
                model = IterVAE(
                    input_dim=x_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=200,
                    output_dim=x_dim,
                    num_encoder_linears=i,
                    num_encoder_iters=1,
                    device=device).to(device)
                
                optimizer = Adam(model.parameters(), lr=lr)
                tlosses, vlosses = train_model(
                    model=model,
                    optimizer=optimizer,
                    epochs=epochs,
                    batch_size=batch_size,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device
                    )
                train_losses_layers.append(tlosses)
                val_losses_layers.append(vlosses)

            for i in range(num_enc):
                model = IterVAE(
                    input_dim=x_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=200,
                    output_dim=x_dim,
                    num_encoder_linears=1,
                    num_encoder_iters=i,
                    device=device).to(device)
                
                optimizer = Adam(model.parameters(), lr=lr)
                tlosses, vlosses = train_model(
                    model=model,
                    optimizer=optimizer,
                    epochs=epochs,
                    batch_size=batch_size,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device
                    )
                train_losses_iters.append(tlosses)
                val_losses_iters.append(vlosses)

            plot_loss(epochs, hidden_dim, train_losses_layers, "Training Loss", "Layers", ds_names[ds])
            plot_loss(epochs, hidden_dim, val_losses_layers, "Validation Loss", "Layers", ds_names[ds])
            plot_loss(epochs, hidden_dim, train_losses_iters, "Training Loss", "Iterations", ds_names[ds])
            plot_loss(epochs, hidden_dim, val_losses_iters, "Validation Loss", "Iterations", ds_names[ds])

if __name__ == "__main__":
    hidden_dims = [200, 400]
    datasets = [EMNIST, CIFAR10, MNIST]
    num_enc = 5
    epochs = 100
    dataset_path = "~/datasets"
    batch_size = 100
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("cuda")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    lr = 1e-3

    train_pipeline(
        hidden_dims,
        datasets,
        num_enc,
        dataset_path,
        batch_size,
        epochs,
        lr,
        device)

            
        