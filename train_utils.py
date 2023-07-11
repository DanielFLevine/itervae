import numpy as np
import torch
import wandb
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from torchvision.models.feature_extraction import create_feature_extractor


# Function to compute the Frechet Inception Distance
def compute_frechet_distance(real_data, generated_data, model, device, batch_size):

    real_activations = []
    generated_activations = []

    pad_vert = (299 - 32) // 2
    pad_horiz = (299 - 32) // 2
    
    for generated_batch in generated_data:
        real_batch = next(real_data)[0].to(device)
        generated_batch = generated_batch.to(device)
        real_batch = nn.functional.interpolate(real_batch, size=(299, 299), mode='bilinear', align_corners=False)
        generated_batch = nn.functional.interpolate(generated_batch, size=(299, 299), mode='bilinear', align_corners=False)
        real_batch = 2 * real_batch - 1  # Scale from range (0, 1) to range (-1, 1)
        generated_batch = 2 * generated_batch - 1  # Scale from range (0, 1) to range (-1, 1)
        # real_batch = nn.functional.pad(real_batch, (pad_horiz, pad_horiz, pad_vert, pad_vert))
        # generated_batch = nn.functional.pad(generated_batch, (pad_horiz, pad_horiz, pad_vert, pad_vert))


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


def get_inception_for_fid(device):

    inception = inception_v3(weights="IMAGENET1K_V1", transform_input=False).to(device).eval()

    return_nodes = {
        "avgpool":"avgpool"
    }

    return create_feature_extractor(inception, return_nodes=return_nodes)


def loss_function(
        x,
        x_hat,
        mean,
        log_var,
        encoder_trajectory,
        decoder_trajectory,
        traj_len,
        beta,
        gamma
        ):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    trajectory_loss = torch.diagonal(torch.cdist(encoder_trajectory, decoder_trajectory, p=2.0))
    mean_trajectory_loss = torch.sum(trajectory_loss)/traj_len
    total_loss = reproduction_loss + (beta*KLD) + (gamma*mean_trajectory_loss)

    return total_loss, reproduction_loss, KLD, mean_trajectory_loss

def train_model(
        model,
        optimizer,
        epochs,
        batch_size,
        beta,
        gamma,
        train_loader,
        test_loader,
        iter_test_loader,
        device
        ):
    
    print("Loading inception model for benchmarking...")
    inception_model = get_inception_for_fid(device)

    print("Start training VAE...")
    train_losses = []
    val_losses = []
    fid_scores = []
    im_shape = (batch_size, 3, 32, 32)
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        train_rep_loss = 0
        train_kld = 0
        train_mean_traj_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()

            x_hat, mean, log_var, encoder_trajectory, decoder_trajectory = model(x)
            total_loss, rep_loss, kld, mean_traj_loss = loss_function(
                x,
                x_hat,
                mean,
                log_var,
                encoder_trajectory,
                decoder_trajectory,
                model.num_iters,
                beta,
                gamma
                )
            
            train_loss += total_loss.item()
            train_rep_loss += rep_loss.item()
            train_kld += kld.item()
            train_mean_traj_loss += mean_traj_loss.item()
            
            total_loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / (batch_idx*batch_size)
        avg_train_rep_loss = train_rep_loss / (batch_idx*batch_size)
        avg_train_kld = train_kld / (batch_idx*batch_size)
        avg_train_mean_traj_loss = train_mean_traj_loss / (batch_idx*batch_size)

        train_losses.append(avg_train_loss)
        model.eval()
        val_loss = 0
        val_rep_loss = 0
        val_kld = 0
        val_mean_traj_loss = 0
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.to(device)

            x_hat, mean, log_var, encoder_trajectory, decoder_trajectory = model(x)
            total_loss, rep_loss, kld, mean_traj_loss = loss_function(
                x,
                x_hat,
                mean,
                log_var,
                encoder_trajectory,
                decoder_trajectory,
                model.num_iters,
                beta,
                gamma
                )
            
            val_loss += total_loss.item()
            val_rep_loss += rep_loss.item()
            val_kld += kld.item()
            val_mean_traj_loss += mean_traj_loss.item()

        avg_val_loss = val_loss / (batch_idx*batch_size)
        avg_val_rep_loss = val_rep_loss / (batch_idx*batch_size)
        avg_val_kld = val_kld / (batch_idx*batch_size)
        avg_val_mean_traj_loss = val_mean_traj_loss / (batch_idx*batch_size)

        val_losses.append(avg_val_loss)
        print(f"Epoch: {epoch+1} || Avg Train Loss: {avg_train_loss:.4f} || Avg Val Loss: {avg_val_loss:.4f}")

        num_samples = 2*batch_size
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
        fid = compute_frechet_distance(iter_test_loader, generated_dataloader, inception_model, device, batch_size)
        fid_scores.append(fid)
        wandb.log({
            "Training Loss": avg_train_loss,
            "Validation Loss": avg_val_loss,
            "200 Samples FID": fid,
            "Training Reproduction Loss": avg_train_rep_loss,
            "Training KLD": avg_train_kld,
            "Training Mean Trajectory Loss": avg_train_mean_traj_loss,
            "Validation Reproduction Loss": avg_val_rep_loss,
            "Validation KLD": avg_val_kld,
            "Validation Mean Trajectory Loss": avg_val_mean_traj_loss
            })
        
    print("Finish!!")
    model.eval()

    print(f"\nENCODER SMOOTHING FACTORS: {[model.Encoder.smooth_factors[i] for i in range(model.num_convs)]}")
    print(f"\nDECODER SMOOTHING FACTORS: {[model.Decoder.smooth_factors[i] for i in range(model.num_convs)]}")
    print(f"\nENCODER ITER SMOOTHING FACTORS: {[model.iter_enc_smooth_factors[i] for i in range(model.num_iters)]}")
    print(f"\nDECODER ITER SMOOTHING FACTORS: {[model.iter_dec_smooth_factors[i] for i in range(model.num_iters)]}")
    return train_losses, val_losses, fid_scores