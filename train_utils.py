import numpy as np
import torch
import wandb
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from torchvision.models import inception_v3

# Function to compute the Frechet Inception Distance
def compute_frechet_distance(real_data, generated_data, model, device):

    real_activations = []
    generated_activations = []

    pad_vert = (299 - 32) // 2
    pad_horiz = (299 - 32) // 2
    
    for real_batch, generated_batch in zip(real_data, generated_data):
        real_batch = real_batch[0].to(device)
        generated_batch = generated_batch.to(device)
        real_batch = nn.functional.pad(real_batch, (pad_horiz, pad_horiz, pad_vert, pad_vert))
        generated_batch = nn.functional.pad(generated_batch, (pad_horiz, pad_horiz, pad_vert, pad_vert))


        with torch.no_grad():
            real_activations.append(model(real_batch).cpu().numpy())
            generated_activations.append(model(generated_batch).cpu().numpy())
    
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

def loss_function(x, x_hat, mean, log_var, encoder_trajectory, decoder_trajectory):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    trajectory_loss = torch.diagonal(torch.cdist(encoder_trajectory, decoder_trajectory, p=2.0))


    return reproduction_loss + KLD + torch.sum(trajectory_loss)

def train_model(
        model,
        optimizer,
        epochs,
        batch_size,
        train_loader,
        test_loader,
        device
        ):
    
    print("Loading inception model for benchmarking...")
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device).eval()
    inception_model.fc = nn.Identity()
    inception_model = nn.Sequential(inception_model, nn.Linear(2048, 512)).to(device)

    print("Start training VAE...")
    train_losses = []
    val_losses = []
    fid_scores = []
    x_dim = model.input_dim
    im_shape = (batch_size, 3, 32, 32)
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var, encoder_trajectory, decoder_trajectory = model(x)
            loss = loss_function(x, x_hat, mean, log_var, encoder_trajectory, decoder_trajectory)
            
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / (batch_idx*batch_size)
        train_losses.append(avg_train_loss)
        model.eval()
        val_loss = 0
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(device)

            x_hat, mean, log_var, encoder_trajectory, decoder_trajectory = model(x)
            loss = loss_function(x, x_hat, mean, log_var, encoder_trajectory, decoder_trajectory)
            val_loss += loss.item()
        avg_val_loss = val_loss / (batch_idx*batch_size)
        val_losses.append(avg_val_loss)
        print(f"Epoch: {epoch+1} || Avg Train Loss: {avg_train_loss:.4f} || Avg Val Loss: {avg_val_loss:.4f}")

        num_samples = len(test_loader)
        print(f"Generating {num_samples} samples...")
        generated_samples = []
        with torch.no_grad():
            for _ in range(num_samples // batch_size):
                noise = torch.randn(batch_size, model.latent_dim, device=device)
                generated_batch = model.decode(noise)
                generated_samples.append(generated_batch.cpu().view(im_shape))
        generated_samples = torch.cat(generated_samples, dim=0)
        generated_dataloader = DataLoader(generated_samples, batch_size=batch_size, shuffle=True, drop_last=False)
        print("Computing FID...")
        fid = compute_frechet_distance(test_loader, generated_dataloader, inception_model, device)
        fid_scores.append(fid)
        wandb.log({
            "Training Loss": avg_train_loss,
            "Validation Loss": avg_val_loss,
            "FID": fid
            })
        
    print("Finish!!")
    model.eval()
    return train_losses, val_losses, fid_scores