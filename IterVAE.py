import torch
import torch.nn as nn


class Embedding(nn.Module):
    # ADD VARYING AMOUNTS OF DROPOUT TO SEE IF PERFORMANCE WILL CHANGE
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


class Projection(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(Projection, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        return self.LeakyReLU(self.FC_hidden(x))


class Decoder(nn.Module):
    def __init__(self, hidden_dim, num_linears):
        super(Decoder, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(num_linears)])
        
        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True
        
    def forward(self, x):
        for linear in self.linears:
            x = self.LeakyReLU(linear(x))
        
        return x
    

class Prediction(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Prediction, self).__init__()

        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.training = True

    def forward(self, x):
        return torch.sigmoid(self.FC_output(x))




class IterVAE(nn.Module):
    def __init__(self,
        input_dim,
        hidden_dim,
        latent_dim,
        output_dim,
        num_linears,
        num_iters,
        device):
        super(IterVAE, self).__init__()

        self.num_iters = num_iters
        self.input_dim=input_dim
        self.latent_dim=latent_dim # Needed for generating samples
        self.Embedding = Embedding(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        self.Encoder = Encoder(
            hidden_dim=hidden_dim,
            num_linears=num_linears
        )
        self.MeanVar = MeanVar(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        self.Projection = Projection(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        self.Decoder = Decoder(
            hidden_dim=hidden_dim,
            num_linears=num_linears
        )
        self.Prediction = Prediction(
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

        encoder_trajectory = [h]
        for _ in range(self.num_iters):
            h = self.Encoder(h)
            encoder_trajectory.append(h)
        
        mean, log_var = self.MeanVar(h)

        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)

        h = self.Projection(z)
        decoder_trajectory = [h]
        for _ in range(self.num_iters):
            h = self.Decoder(h)
            decoder_trajectory.append(h)

        x_hat = self.Prediction(h)

        decoder_trajectory.reverse()
        
        return x_hat, mean, log_var, torch.stack(encoder_trajectory), torch.stack(decoder_trajectory)
    
    def decode(self, z):
        h = self.Projection(z)
        for _ in range(self.num_iters):
            h = self.Decoder(h)
        x_hat = self.Prediction(h)
        return x_hat

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