import torch
from tqdm import tqdm
import torch.nn as nn

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def train_model(
        model,
        optimizer,
        epochs,
        batch_size,
        train_loader,
        test_loader,
        device
        ):
    print("Start training VAE...")
    train_losses = []
    val_losses = []
    x_dim = model.input_dim
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
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

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            val_loss += loss.item()
        avg_val_loss = val_loss / (batch_idx*batch_size)
        val_losses.append(avg_val_loss)
        print(f"Epoch: {epoch+1} || Avg Train Loss: {avg_train_loss:.4f} || Avg Val Loss: {avg_val_loss:.4f}")

        
    print("Finish!!")
    model.eval()
    return train_losses, val_losses