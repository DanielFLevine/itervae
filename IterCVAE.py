import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()

        self.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h = self.LeakyReLU(self.Conv2d(x))
        
        return h

class InputConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super(InputConv, self).__init__()
        self.CNN_input = ConvBlock(in_channels, hidden_channels, kernel_size)

        self.training = True
        
    def forward(self, x):
        return self.CNN_input(x)


class Encoder(nn.Module):
    
    def __init__(self, hidden_channels, kernel_size, num_convs, smooth=True):
        super(Encoder, self).__init__()
        self.Convs = nn.ModuleList([ConvBlock(hidden_channels, hidden_channels, kernel_size) for _ in range(num_convs)])
        if smooth:
            self.smooth_factors = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(num_convs)])
        else:
            self.smooth_factors = nn.ParameterList([nn.Parameter(torch.tensor(0.0), requires_grad=False) for _ in range(num_convs)])
        self.num_convs = num_convs
        self.training = True
        
    def forward(self, x):
        for i in range(self.num_convs):
            x = self.smooth_factors[i]*x + (1-self.smooth_factors[i])*self.Convs[i](x)
        return x


class MeanVar(nn.Module):

    def __init__(self, hidden_dim, latent_dim):
        super(MeanVar, self).__init__()

        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        
        self.training = True
        
    def forward(self, x):
        mean = self.FC_mean(x)
        log_var = self.FC_var(x)                     
        
        return mean, log_var


class Projection(nn.Module):
    def __init__(self, latent_dim, hidden_dim, hidden_channels):
        super(Projection, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.hidden_channels = hidden_channels

        self.training = True

    def forward(self, x):
        return self.LeakyReLU(self.FC_hidden(x)).view(-1, self.hidden_channels, 32, 32)


class Decoder(nn.Module):
    def __init__(self, hidden_channels, kernel_size, num_convs, smooth=True):
        super(Decoder, self).__init__()
        self.Convs = nn.ModuleList([ConvBlock(hidden_channels, hidden_channels, kernel_size) for _ in range(num_convs)])
        if smooth:
            self.smooth_factors = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(num_convs)])
        else:
            self.smooth_factors = nn.ParameterList([nn.Parameter(torch.tensor(0.0), requires_grad=False) for _ in range(num_convs)])
        self.num_convs = num_convs
        self.training = True
        
    def forward(self, x):
        for i in range(self.num_convs):
            x = self.smooth_factors[i]*x + (1-self.smooth_factors[i])*self.Convs[i](x)
        return x
    

class Prediction(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super(Prediction, self).__init__()
        self.CNN_output = ConvBlock(hidden_channels, in_channels, kernel_size)

        self.training = True
        
    def forward(self, x):
        return torch.sigmoid(self.CNN_output(x))




class IterCVAE(nn.Module):
    def __init__(self,
        in_channels,
        hidden_channels,
        hidden_dim,
        latent_dim,
        kernel_size,
        num_convs,
        num_iters,
        device):
        super(IterCVAE, self).__init__()
        self.smooth = True if num_iters==1 else False
        self.num_iters = num_iters
        self.num_convs = num_convs
        self.hidden_dim = hidden_dim
        self.latent_dim=latent_dim # Needed for generating samples
        self.InputConv = InputConv(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size
        )
        self.Encoder = Encoder(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_convs=num_convs,
            smooth=self.smooth
        )
        self.MeanVar = MeanVar(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        self.Projection = Projection(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            hidden_channels=hidden_channels
        )
        self.Decoder = Decoder(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_convs=num_convs,
            smooth=self.smooth
        )
        self.Prediction = Prediction(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size
        )
        if self.num_iters > 1:
            self.iter_enc_smooth_factors = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(num_iters)])
            self.iter_dec_smooth_factors = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(num_iters)])
        else:
            self.iter_enc_smooth_factors = nn.ParameterList([nn.Parameter(torch.tensor(0.0), requires_grad=False) for _ in range(num_iters)])
            self.iter_dec_smooth_factors = nn.ParameterList([nn.Parameter(torch.tensor(0.0), requires_grad=False) for _ in range(num_iters)])
        self.device = device
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):

        h = self.InputConv(x)
        encoder_trajectory = [h.view(-1, self.hidden_dim)]
        for i in range(self.num_iters):
            h = (self.iter_enc_smooth_factors[i]*h) + ((1-self.iter_enc_smooth_factors[i])*self.Encoder(h))
            encoder_trajectory.append(h.view(-1, self.hidden_dim))
        
        mean, log_var = self.MeanVar(h.view(-1, self.hidden_dim))

        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)

        h = self.Projection(z)
        decoder_trajectory = [h.view(-1, self.hidden_dim)]
        for i in range(self.num_iters):
            h = (self.iter_dec_smooth_factors[i]*h) + ((1-self.iter_dec_smooth_factors[i])*self.Decoder(h))
            decoder_trajectory.append(h.view(-1, self.hidden_dim))
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
    pass