import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim=64):
        super(MLP, self).__init__()
        '''
        layer_sizes: list of input sizes: forward/inverse model: 3 layers with 64 nodes in each layer
        '''

        self.net = nn.Sequential(*[nn.Linear(input_size, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, output_size)])

    def forward(self, x, y):
        return self.net(x)

class MLP_sigmoid(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim=64):
        super(MLP_sigmoid, self).__init__()
        '''
        layer_sizes: list of input sizes: forward/inverse model: 3 layers with 64 nodes in each layer
        '''

        self.net = nn.Sequential(*[nn.Linear(input_size, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, output_size),
                                   nn.Sigmoid()])

    def forward(self, x, y):
        return self.net(x)

class MLP_ReLu(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim=64):
        super(MLP_ReLu, self).__init__()
        '''
        layer_sizes: list of input sizes: forward/inverse model: 3 layers with 64 nodes in each layer
        '''

        self.net = nn.Sequential(*[nn.Linear(input_size, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, output_size),
                                   nn.ReLU()])

    def forward(self, x, y):
        return self.net(x)


class MLP_4(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim=320):
        super(MLP_4, self).__init__()
        '''
        layer_sizes: list of input sizes: forward/inverse model: 4 layers with 320 nodes in each layer
        '''

        self.net = nn.Sequential(*[nn.Linear(input_size, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, output_size)])

    def forward(self, x, y):
        return self.net(x)

class MLP_sigmoid_4(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim=320):
        super(MLP_sigmoid_4, self).__init__()
        '''
        layer_sizes: list of input sizes: forward/inverse model: 4 layers with 320 nodes in each layer
        '''

        self.net = nn.Sequential(*[nn.Linear(input_size, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, output_size),
                                   nn.Sigmoid()])

    def forward(self, x, y):
        return self.net(x)

class MLP_ReLu_4(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim=320):
        super(MLP_ReLu_4, self).__init__()
        '''
        layer_sizes: list of input sizes: forward/inverse model: 4 layers with 320 nodes in each layer
        '''

        self.net = nn.Sequential(*[nn.Linear(input_size, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, output_size),
                                   nn.ReLU()])

    def forward(self, x, y):
        return self.net(x)


class MLP_deep(nn.Module):

    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        '''
        layer_sizes: list of input sizes: forward/inverse model: 4 layers with 320 nodes in each layer
        '''

        self.net = nn.Sequential(*[nn.Linear(input_size, 320),
                                   nn.ReLU(),
                                   #nn.BatchNorm1d(320),
                                   nn.Linear(320, 320),
                                   nn.ReLU(),
                                   #nn.BatchNorm1d(320),
                                   nn.Linear(320, 320),
                                   nn.ReLU(),
                                   #nn.BatchNorm1d(320),
                                   nn.Linear(320, 320),
                                   nn.ReLU(),
                                   nn.Linear(320, output_size)])

    def forward(self, x, y):
        return self.net(x)

class TandemNet(nn.Module):

    def __init__(self, forward_model, inverse_model):
        super(TandemNet, self).__init__()
        self.forward_model = forward_model
        self.inverse_model = inverse_model

    def forward(self, x, y):
        # x: structure, y: CIE coordinate out: cie
        '''
        Pass the desired target x to the tandem network.
        '''

        pred = self.inverse_model(y, x)
        out = self.forward_model(pred, x)

        return out

    def pred(self, y):
        pred = self.inverse_model(y, None)
        # pred : structure
        return pred


class cVAE(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_dim=256, forward_dim=3):
        super(cVAE, self).__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim

        # encoder
        self.encoder = nn.Sequential(*[nn.Linear(input_size, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU()])

        self.forward_net = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(hidden_dim),
                                           nn.Linear(hidden_dim, forward_dim)])

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.decoder = nn.Sequential(*[nn.Linear(latent_dim + forward_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, input_size)])

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y):
        recon_x = self.decoder(torch.cat((z, y), dim=1))
        return recon_x

    def forward(self, x, y):
        h = self.encoder(x)
        y_pred = self.forward_net(h)

        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = self.reparameterize(mu, logvar)

        return self.decode(z, y), mu, logvar, y_pred

class cVAE_Full(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_dim=256, forward_dim=3):
        super(cVAE_Full, self).__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim

        # encoder
        self.encoder = nn.Sequential(*[nn.Linear(input_size+forward_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU()])
        
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.decoder = nn.Sequential(*[nn.Linear(latent_dim + forward_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, input_size)])

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y):
        recon_x = self.decoder(torch.cat((z, y), dim=1))
        return recon_x

    def forward(self, x, y):
        y_pred = y
        h = self.encoder(torch.cat((x, y), dim=1))

        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = self.reparameterize(mu, logvar)

        return self.decode(z, y), mu, logvar, y_pred
    
    def inference(self, y):
        
        mu, logvar = torch.zeros([y.size()[0], self.latent_dim]), torch.zeros([y.size()[0], self.latent_dim])
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar, y


class cVAE_hybrid(nn.Module):

    def __init__(self, forward_model, vae_model):
        super(cVAE_hybrid, self).__init__()
        self.forward_model = forward_model
        self.vae_model = vae_model

    def forward(self, x, y):
        # the prediction is based on cVAE_GSNN model
        '''
        Pass the desired target x to the vae_hybrid network.
        '''
        
        x_pred, mu, logvar, x_hat = self.vae_model(x, y)
        
        y_pred = self.forward_model(x_pred, None)
        return x_pred, mu, logvar, x_hat, y_pred 

    def pred(self, x):
        pred = self.forward_model(x, None)
        return pred


class cVAE_GSNN(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_dim=256, forward_dim=3):
        super(cVAE_GSNN, self).__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim

        # encoder
        self.encoder = nn.Sequential(*[nn.Linear(input_size+forward_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU()])


        self.forward_net = nn.Sequential(*[nn.Linear(forward_dim, 64),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(64),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(64),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, input_size)])
        
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.decoder = nn.Sequential(*[nn.Linear(latent_dim + forward_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, input_size)])

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y):
        temp = torch.cat((z,y), dim=1)
        recon_x = self.decoder(torch.cat((z, y), dim=1))
        return recon_x

    def forward(self, x, y):
        x_hat = self.forward_net(y)
        h = self.encoder(torch.cat((x, y), dim=1))

        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = self.reparameterize(mu, logvar)
        
        return self.decode(z, y), mu, logvar, x_hat
    
    def inference(self, y):
        x = self.forward_net(y)
        h = self.encoder(torch.cat((x, y), dim=1))
        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = self.reparameterize(mu, logvar)

        return self.decode(z, y), mu, logvar, x
    
class cVAE_GSNN1(nn.Module):
    # a deeper but narrow network
    def __init__(self, input_size, latent_dim, hidden_dim=64, forward_dim=3):
        super(cVAE_GSNN1, self).__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim

        # encoder
        self.encoder = nn.Sequential(*[nn.Linear(input_size+forward_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim)])


        self.forward_net = nn.Sequential(*[nn.Linear(forward_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, input_size)])
        
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.decoder = nn.Sequential(*[nn.Linear(latent_dim + forward_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, input_size)])

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y):
        temp = torch.cat((z,y), dim=1)
        recon_x = self.decoder(torch.cat((z, y), dim=1))
        return recon_x

    def forward(self, x, y):
        x_hat = self.forward_net(y)
        h = self.encoder(torch.cat((x, y), dim=1))

        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = self.reparameterize(mu, logvar)
        
        return self.decode(z, y), mu, logvar, x_hat
    
    def inference(self, y):
        x = self.forward_net(y)
        h = self.encoder(torch.cat((x, y), dim=1))
        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = self.reparameterize(mu, logvar)

        return self.decode(z, y), mu, logvar, x


class cVAE_tandem(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_dim=256, forward_dim=3):
        super(cVAE_tandem, self).__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim

        # encoder
        self.encoder = nn.Sequential(*[nn.Linear(input_size+forward_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU()])


        self.forward_net = nn.Sequential(*[nn.Linear(forward_dim, 64),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(64),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(64),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, input_size)])
        
        self.inverse_net = nn.Sequential(*[nn.Linear(input_size, 64),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(64),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(64),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, forward_dim)])
        
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.decoder = nn.Sequential(*[nn.Linear(latent_dim + forward_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, input_size)])

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y):
        temp = torch.cat((z,y), dim=1)
        recon_x = self.decoder(torch.cat((z, y), dim=1))
        return recon_x

    def forward(self, x, y):
        x_pred = self.forward_net(y)
        y_pred= self.inverse_net(x_pred)
        
        h = self.encoder(torch.cat((x, y), dim=1))

        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = self.reparameterize(mu, logvar)
        
        return self.decode(z, y), mu, logvar, x_pred, y_pred
    
    def inference(self, y):
        x = self.forward_net(y)
        h = self.encoder(torch.cat((x, y), dim=1))
        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = self.reparameterize(mu, logvar)

        return self.decode(z, y), mu, logvar, x


class cVAE_new(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_dim=256, forward_dim=3):
        super(cVAE_new, self).__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim

        # encoder
        self.encoder = nn.Sequential(*[nn.Linear(input_size+forward_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU()])


        self.forward_net = nn.Sequential(*[nn.Linear(input_size, 64),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(64),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(64),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, forward_dim)])
        
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.decoder = nn.Sequential(*[nn.Linear(latent_dim + forward_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, input_size)])

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y):
        recon_x = self.decoder(torch.cat((z, y), dim=1))
        return recon_x

    def forward(self, x, y):
        y_pred = self.forward_net(x)
        h = self.encoder(torch.cat((x, y_pred), dim=1))

        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = self.reparameterize(mu, logvar)

        return self.decode(z, y), mu, logvar, y_pred
# conditional GAN


class Generator(nn.Module):
    def __init__(self, input_size, output_size, noise_dim=3, hidden_dim=64):
        super(Generator, self).__init__()

        self.input_size = input_size

        self.net = nn.Sequential(*[nn.Linear(input_size + noise_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, output_size)])

    def forward(self, x, noise):
        y = self.net(torch.cat((x, noise), dim=1))
        return y

class Discriminator(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=64):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(*[nn.Linear(input_size+output_size, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   #nn.BatchNorm1d(hidden_dim), #
                                   #don't use batch norm for the D input layer and G output layer to aviod the oscillation and model instability 
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(0.2)])
        

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        # self.aux_layer = nn.Sequential(nn.Linear(128, 3))

    def forward(self, y_fake, x):
        h = self.net(torch.cat((y_fake, x), dim=1))
        validity = self.adv_layer(h)
        # label = self.aux_layer(h)

        return validity

class cGAN(nn.Module):
    def __init__(self, input_size, output_size, noise_dim=3, hidden_dim=64):
        super(cGAN, self).__init__()

        self.generator = Generator(
            input_size, output_size, noise_dim=noise_dim, hidden_dim=hidden_dim)
        self.discriminator = Discriminator(
            output_size, input_size, hidden_dim=hidden_dim)

        self.noise_dim = noise_dim

    def forward(self, x, noise):

        y_fake = self.generator(x, noise)
        validity = self.discriminator(y_fake, x)

        return validity

    def sample_noise(self, batch_size, prior=1):

        if prior == 1:
            z = torch.tensor(np.random.normal(0, 1, (batch_size, self.noise_dim))).float()
        else:
            z = torch.tensor(np.random.uniform(0, 1, (batch_size, self.noise_dim))).float()
        return z

    def sample_noise_M(self, batch_size):
        M = 100
        z = torch.tensor(np.random.normal(
            0, 1, (batch_size*M, self.noise_dim))).float()
        return z

# invertible neural network

class INN(nn.Module):

    def __init__(self, ndim_total, dim_x, dim_y, dim_z, hidden_dim = 128):
        super(INN, self).__init__()

        nodes = [InputNode(ndim_total, name = 'input')]
        self.hidden_dim = hidden_dim
        self.ndim_total = ndim_total
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z

        for k in range(4):
            nodes.append(Node(nodes[-1],
                                GLOWCouplingBlock,
                                {'subnet_constructor': self.subnet_fc, 'clamp': 2.0},
                                name=F'coupling_{k}'))

            nodes.append(Node(nodes[-1],
                            PermuteRandom,
                            {'seed': k},
                            name=F'permute_{k}'))

        nodes.append(OutputNode(nodes[-1], name = 'output'))

        self.model = ReversibleGraphNet(nodes, verbose = False)
        self.zeros_noise_scale = 5e-2
        self.y_noise_scale = 1e-1

    def forward(self, x, rev=False):
        return self.model(x, rev=rev)

    def subnet_fc(self, c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, self.hidden_dim),
                             nn.ReLU(),
                             nn.Linear(self.hidden_dim, c_out))

    def create_padding(self, batch_size):

        pad_x = self.zeros_noise_scale * torch.randn(batch_size, self.ndim_total - self.dim_x)
        pad_yz = self.zeros_noise_scale * torch.randn(batch_size, self.ndim_total - self.dim_y - self.dim_z)

        return pad_x, pad_yz

    # def add_noise_y(self, y, batch_size):

    #     y += self.y_noise_scale * torch.randn(batch_size, ndim_y, dtype=torch.float, device=device)


class Discriminator_old(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=64):
        super(Discriminator, self).__init__()

        self.net1 = nn.Sequential(*[nn.Linear(input_size+output_size, hidden_dim),
                                   nn.ReLU(),
                                   #nn.BatchNorm1d(hidden_dim), #
                                   #don't use batch norm for the D input layer and G output layer to aviod the oscillation and model instability 
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU()])
        

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        # self.aux_layer = nn.Sequential(nn.Linear(128, 3))

    def forward(self, y_fake, x):
        h = self.net(torch.cat((y_fake, x), dim=1))
        validity = self.adv_layer(h)
        # label = self.aux_layer(h)

        return validity

class Generator_old(nn.Module):
    def __init__(self, input_size, output_size, noise_dim=3, hidden_dim=64):
        super(Generator, self).__init__()

        self.input_size = input_size

        self.net = nn.Sequential(*[nn.Linear(input_size + noise_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, output_size)],
                                   nn.ReLU())

    def forward(self, x, noise):
        y = self.net(torch.cat((x, noise), dim=1))
        return y
