import torch
import torch.nn as nn


class Embedding(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super(Embedding, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h = self.LeakyReLU(self.FC_input(x))
        
        return h


class Encoder(nn.Module):
    
    def __init__(self, hidden_dim, num_linears):
        super(Encoder, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(num_linears)])
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        for linear in self.linears:
            x = self.LeakyReLU(linear(x))
        return x


class MeanVar(nn.Module):

    def __init__(self, hidden_dim, latent_dim):
        super(MeanVar, self).__init__()

        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear (hidden_dim, latent_dim)
        
        self.training = True
        
    def forward(self, x):
        mean = self.FC_mean(x)
        log_var = self.FC_var(x)                     
        
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class IterVAE(nn.Module):
    def __init__(self,
        input_dim,
        hidden_dim,
        latent_dim,
        output_dim,
        num_encoder_linears,
        num_encoder_iters,
        device):
        super(IterVAE, self).__init__()
        self.num_encoder_iters = num_encoder_iters
        self.input_dim=input_dim
        self.Embedding = Embedding(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        self.Encoder = Encoder(
            hidden_dim=hidden_dim,
            num_linears=num_encoder_linears
        )
        self.MeanVar = MeanVar(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        self.Decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        self.device = device
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        h = self.Embedding(x)
        for iter in range(self.num_encoder_iters):
            h = self.Encoder(h)
        mean, log_var = self.MeanVar(h)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var

if __name__ == "__main__":
    model = IterVAE(
        100,
        200,
        200,
        100,
        1,
        1,
        torch.device("cpu")
    )