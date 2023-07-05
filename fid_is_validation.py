import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np


# Function to compute the inception score
def compute_inception_score(data, batch_size, model, device):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    
    softmax = nn.Softmax(dim=1)
    
    scores = []
    
    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            preds = model(batch)
            preds = softmax(preds)
        
        scores.append(preds.cpu().numpy())
    
    preds = np.concatenate(scores, axis=0)
    scores = np.mean(preds, axis=0)
    kl_divs = preds * (np.log(preds) - np.log(np.expand_dims(scores, 0)))
    kl_divs = np.mean(np.sum(kl_divs, axis=1))
    
    inception_score = np.exp(kl_divs)
    
    return inception_score

# Function to compute the Frechet Inception Distance
def compute_frechet_distance(real_data, generated_data, batch_size, model, device):
    real_dataloader = DataLoader(real_data, batch_size=batch_size, shuffle=True, drop_last=True)
    generated_dataloader = DataLoader(generated_data, batch_size=batch_size, shuffle=True, drop_last=True)

    
    real_activations = []
    generated_activations = []
    
    for real_batch, generated_batch in zip(real_dataloader, generated_dataloader):
        real_batch = real_batch.to(device)
        generated_batch = generated_batch.to(device)
        
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

# Example usage
def main():
    # Set the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the batch size and number of samples
    batch_size = 64
    num_samples = 10000

    # Load the pretrained Inception-v3 network
    model = inception_v3(pretrained=True, transform_input=False).to(device).eval()
    model.fc = nn.Identity()
    model = nn.Sequential(model, nn.Linear(2048, 512)).to(device)

    # Define the data transformations
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the real data
    real_data = CIFAR10(root="./data", download=True, train=True, transform=transform)
    real_data = real_data.data[:num_samples]

    # Generate fake data (replace with your own generator model)
    generator = YourGeneratorModel().to(device).eval()
    fake_data = []
    with torch.no_grad():
        for _ in range(num_samples // batch_size):
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_batch = generator(noise)
            fake_data.append(fake_batch.cpu())
    fake_data = torch.cat(fake_data, dim=0)

    # Compute Frechet Inception Distance
    fid = compute_frechet_distance(real_data, fake_data, batch_size, model, device)
    print("Frechet Inception Distance:", fid)

    # Compute Inception Score
    iscore = compute_inception_score(fake_data, batch_size, model, device)
    print("Inception Score:", iscore)

if __name__ == "__main__":
    main()
