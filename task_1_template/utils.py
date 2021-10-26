import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn import functional as F
from models import MLP, TandemNet, cVAE, cGAN, INN, cVAE_new, cVAE_GSNN, cVAE_Full, cVAE_tandem, cVAE_hybrid

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# class Trainer():

#     def __init__(self,
#                  model,
#                  optimizer,
#                  train_loader,
#                  val_loader,
#                  test_loader,
#                  criterion,
#                  epochs,
#                  model_name):

#         self.model_name = model_name
#         self.model = model
#         self.optimizer = optimizer
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.test_loader = test_loader
#         self.criterion = criterion
#         self.epochs = epochs

#         self.path = './models/' + model_name + '.pth'

#     def train(self):
#         self.model.train()
#         loss_epoch = 0
#         for x, y in self.train_loader:

#             self.optimizer.zero_grad()
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             pred = self.model(x, y)
#             loss = self.get_loss(x, y, pred)
#             loss.backward()
#             self.optimizer.step()
#             loss_epoch += loss.to('cpu').item() * len(x)

#         return loss_epoch / len(self.train_loader.dataset)

#     def val(self):
#         self.model.eval()
#         loss_epoch = 0
#         with torch.no_grad():
#             for x, y in self.val_loader:

#                 x, y = x.to(DEVICE), y.to(DEVICE)
#                 pred = self.model(x, y)
#                 loss = self.get_loss(x, y, pred)
#                 loss_epoch += loss.to('cpu').item() * len(x)

#         return loss_epoch / len(self.val_loader.dataset)

#     def test(self):
#         self.model.eval()
#         loss_epoch = 0
#         with torch.no_grad():
#             for x, y in self.test_loader:

#                 x, y = x.to(DEVICE), y.to(DEVICE)
#                 pred = self.model(x, y)
#                 loss = self.get_loss(x, y, pred)
#                 loss_epoch += loss.to('cpu').item() * len(x)

#         return loss_epoch / len(self.test_loader.dataset)

#     def fit(self):

#         for e in range(self.epochs):

#             loss_train = self.train()
#             loss_val = self.val()
#             print('Epoch {}, train loss {:.3f}, val loss {:.3f}'.format(
#                 e, loss_train, loss_val))

#         loss_test = self.test()
#         print('Training finished! Test loss {:.3f}'.format(loss_test))

#         self.save_checkpoint(e, loss_test)
#         print('Saved final trained model.')

#     def save_checkpoint(self, epoch, loss):
#         # torch.save(self.model.state_dict(), './models/'+filename)

#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'loss': loss,
#         }, self.path)

#     def load_checkpoint(self):

#         checkpoint = torch.load(self.path)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         epoch = checkpoint['epoch']
#         loss = checkpoint['loss']

#         print("Loaded model, epoch {}, loss {}..".format(epoch, loss))

#     def get_loss(self, x, y, pred):
#         '''
#         Loss for training simple forward and inverse networks.
#         '''
        
#         if self.model_name in ['inverse_model', 'forward_model']:
#             return self.criterion(pred, y)
#         elif self.model_name in ['tandem_net']:
#             return self.criterion(pred, x)
#         elif self.model_name == 'vae':
#             # see Appendix B from VAE paper:
#             # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#             # https://arxiv.org/abs/1312.6114
#             # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#             recon_x, mu, logvar, y_pred = pred
#             recon_loss = self.criterion(recon_x, x)
#             KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#             pred_loss = self.criterion(y_pred, y)

#             # print(BCE, KLD)
#             return recon_loss + KLD + pred_loss
#         else:
#             raise NotImplementedError


def evaluate_tandem_accuracy(model, dataset):
    '''
    returns:
        x_raw: original desired xyY
        x_raw_pred: xyY predicted by the forward module for the inversely designed structure
        y_raw: original structure parameters
        y_raw_pred: inversely designed parameters.
    '''
    model.eval()
    with torch.no_grad():
        mean, std = torch.tensor(dataset.scaler.mean_).to(DEVICE), torch.tensor(np.sqrt(dataset.scaler.var_)).to(DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        # get MSE for the design
        y_pred = model.pred(x)
        rmse = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        print("Tandem net Design RMSE loss {:.3f}".format(rmse.item()))

        # get RMSE
        y_pred_raw = y_pred * std[x_dim:] + mean[x_dim:]
        y_raw= y * std[x_dim:] + mean[x_dim:]
        rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
        print('Tandem Design RMSE loss {:.3f}'.format(rmse_design_raw.item()))

        # get difference between the obtained CIE and the actual target CIE
        x_pred = model(x, y)
        rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss {:.3f}'.format(rmse_cie))

        # compare differnet between the obtained CIE and the actual target CIE in the original space
        x_pred_raw = x_pred * std[:x_dim] + mean[:x_dim]
        x_raw = x * std[:x_dim] + mean[:x_dim]
        rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()

# def evaluate_tandem_minmax_accuracy(model, dataset, show=1):
#     '''
#     returns:
#         x_raw: original desired xyY
#         x_raw_pred: xyY predicted by the forward module for the inversely designed structure
#         y_raw: original structure parameters
#         y_raw_pred: inversely designed parameters.
#     '''
#     model.eval()
#     with torch.no_grad():
#         range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)
#         x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
#         x_dim = x.size()[1]

#         # get MSE for the design
#         y_pred = model.pred(x)
#         rmse = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
#         if show==1:
#             print("Tandem net Design RMSE loss {:.3f}".format(rmse.item()))

#         # get RMSE
#         y_pred_raw = y_pred *range_[x_dim:] +min_[x_dim:]
#         y_raw =  y *range_[x_dim:] +min_[x_dim:]
#         rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
#         if show==1:
#             print('Tandem Design RMSE loss {:.3f}'.format(rmse_design_raw.item()))

#         # get difference between the obtained CIE and the actual target CIE
#         x_pred = model(x, y)
#         rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
#         if show==1:
#             print('Reconstruct RMSE loss {:.3f}'.format(rmse_cie))

#         # compare differnet between the obtained CIE and the actual target CIE in the original space
#         x_pred_raw = x_pred *range_[:x_dim] + min_[:x_dim]
#         x_raw =  x *range_[:x_dim] +min_[:x_dim]   
        
#         rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
#         if show==1:
#             print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

#     return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()

# def evaluate_tandem_minmax_accuracy(model, dataset,forward_model, show=1):
#     # evaluate the tandem prediction using any desired forward model
#     '''
#     returns:
#         x_raw: original desired xyY
#         x_raw_pred: xyY predicted by the forward module for the inversely designed structure
#         y_raw: original structure parameters
#         y_raw_pred: inversely designed parameters.
#     '''
#     model.eval()
#     forward_model.eval()
#     with torch.no_grad():
#         range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)
#         x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
#         x_dim = x.size()[1]

#         # get MSE for the design
#         y_pred = model.pred(x)
#         x_pred = forward_model(y_pred, None)
        
#         rmse = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
#         if show==1:
#             print("Tandem net Design RMSE loss {:.3f}".format(rmse.item()))

#         # get RMSE
#         y_pred_raw = y_pred *range_[x_dim:] +min_[x_dim:]
#         y_raw =  y *range_[x_dim:] +min_[x_dim:]
#         rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
#         if show==1:
#             print('Tandem Design RMSE loss {:.3f}'.format(rmse_design_raw.item()))

#         # get difference between the obtained CIE and the actual target CIE
#         #x_pred = model(x, y)
#         rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
#         if show==1:
#             print('Reconstruct RMSE loss {:.3f}'.format(rmse_cie))

#         # compare differnet between the obtained CIE and the actual target CIE in the original space
#         x_pred_raw = x_pred *range_[:x_dim] + min_[:x_dim]
#         x_raw =  x *range_[:x_dim] +min_[:x_dim]   
        
#         rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
#         if show==1:
#             print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

#     return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()

def evaluate_simple_inverse(forward_model, inverse_model, dataset):

    inverse_model.eval()
    with torch.no_grad():
        mean, std = torch.tensor(dataset.scaler.mean_).to(DEVICE), torch.tensor(np.sqrt(dataset.scaler.var_)).to(DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        # get MSE for the design
        y_pred = inverse_model(x, y)
        rmse = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        print("Simple net Design RMSE loss {:.3f}".format(rmse.item()))

        # get RMSE
        y_pred_raw = y_pred * std[x_dim:] + mean[x_dim:]
        y_raw= y * std[x_dim:] + mean[x_dim:]
        rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
        print('Simple net RMSE loss {:.3f}'.format(rmse_design_raw.item()))

        # get difference between the obtained CIE and the actual target CIE
        x_pred = forward_model(y_pred, y)
        rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss {:.3f}'.format(rmse_cie))

        # compare differnet between the obtained CIE and the actual target CIE in the original space
        x_pred_raw = x_pred * std[:x_dim] + mean[:x_dim]
        x_raw = x * std[:x_dim] + mean[:x_dim]
        rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()

def evaluate_vae_inverse(forward_model, vae_model, configs, dataset):

    vae_model.eval()
    with torch.no_grad():
        mean, std = torch.tensor(dataset.scaler.mean_).to(DEVICE), torch.tensor(np.sqrt(dataset.scaler.var_)).to(DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        mu, logvar = torch.zeros((len(x), configs['latent_dim'])), torch.zeros((len(x), configs['latent_dim']))   #why 0, 0?
        z = vae_model.reparameterize(mu, logvar).to(DEVICE)
        y_pred = vae_model.decode(z, x)

        # get MSE for the design
        rmse = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        print("Simple net Design RMSE loss {:.3f}".format(rmse.item()))

        # get RMSE
        y_pred_raw = y_pred * std[x_dim:] + mean[x_dim:]
        y_raw= y * std[x_dim:] + mean[x_dim:]
        rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
        print('Simple net RMSE loss {:.3f}'.format(rmse_design_raw.item()))

        # get difference between the obtained CIE and the actual target CIE
        x_pred = forward_model(y_pred, y)
        rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss {:.3f}'.format(rmse_cie))

        # compare differnet between the obtained CIE and the actual target CIE in the original space
        x_pred_raw = x_pred * std[:x_dim] + mean[:x_dim]
        x_raw = x * std[:x_dim] + mean[:x_dim]
        rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()

# def evaluate_vae_minmax_inverse(forward_model, vae_model, configs, dataset, show=1):

#     vae_model.eval()
#     with torch.no_grad():
#         range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)
#         x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
#         x_dim = x.size()[1]

#         mu, logvar = torch.zeros((len(x), configs['latent_dim'])), torch.zeros((len(x), configs['latent_dim']))
#         z = vae_model.reparameterize(mu, logvar).to(DEVICE)
#         y_pred = vae_model.decode(z, x)

#         # get MSE for the design
#         rmse = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
#         if show==1:
#             print("Simple net Design RMSE loss {:.3f}".format(rmse.item()))

#         # get RMSE
#         y_pred_raw = y_pred *range_[x_dim:] +min_[x_dim:]
#         y_raw =  y *range_[x_dim:] +min_[x_dim:]
#         rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
#         if show==1:
#             print('Simple net RMSE loss {:.3f}'.format(rmse_design_raw.item()))

#         # get difference between the obtained CIE and the actual target CIE
#         x_pred = forward_model(y_pred, y)
#         rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
#         if show==1:
#             print('Reconstruct RMSE loss {:.3f}'.format(rmse_cie))

#         # compare differnet between the obtained CIE and the actual target CIE in the original space
#         x_pred_raw = x_pred *range_[:x_dim] + min_[:x_dim]
#         x_raw =  x *range_[:x_dim] +min_[:x_dim]        
#         rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
#         if show==1:
#             print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

#     return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()


# def evaluate_vae_hybrid_minmax_inverse(forward_model, vae_model, configs, dataset, show=1):

#     vae_model.eval()
#     with torch.no_grad():
#         range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)
#         x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
#         x_dim = x.size()[1]

#         mu, logvar = torch.zeros((len(x), configs['latent_dim'])), torch.zeros((len(x), configs['latent_dim']))
#         z = vae_model.vae_model.reparameterize(mu, logvar).to(DEVICE)
#         y_pred = vae_model.vae_model.decode(z, x)

#         # get MSE for the design
#         rmse = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
#         if show==1:
#             print("Simple net Design RMSE loss {:.3f}".format(rmse.item()))

#         # get RMSE
#         y_pred_raw = y_pred *range_[x_dim:] +min_[x_dim:]
#         y_raw =  y *range_[x_dim:] +min_[x_dim:]
#         rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
#         if show==1:
#             print('Simple net RMSE loss {:.3f}'.format(rmse_design_raw.item()))

#         # get difference between the obtained CIE and the actual target CIE
#         x_pred = forward_model(y_pred, y)
#         rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
#         if show==1:
#             print('Reconstruct RMSE loss {:.3f}'.format(rmse_cie))

#         # compare differnet between the obtained CIE and the actual target CIE in the original space
#         x_pred_raw = x_pred *range_[:x_dim] + min_[:x_dim]
#         x_raw =  x *range_[:x_dim] +min_[:x_dim]        
#         rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
#         if show==1:
#             print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

#     return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()

# def evaluate_vae_minmax_GSNN_inverse(forward_model, vae_model, configs, dataset, show=1):

#     vae_model.eval()
#     with torch.no_grad():
#         range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)
#         x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
#         x_dim = x.size()[1]

#         y_pred, mu, logvar, temp = vae_model.inference(x)

#         # get MSE for the design
#         rmse = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
#         if show==1:
#             print("Simple net Design RMSE loss {:.3f}".format(rmse.item()))

#         # get RMSE
#         y_pred_raw = y_pred *range_[x_dim:] +min_[x_dim:]
#         y_raw =  y *range_[x_dim:] +min_[x_dim:]
#         rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
#         if show==1:
#             print('Simple net RMSE loss {:.3f}'.format(rmse_design_raw.item()))

#         # get difference between the obtained CIE and the actual target CIE
#         x_pred = forward_model(y_pred, y)
#         rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
#         if show==1:
#             print('Reconstruct RMSE loss {:.3f}'.format(rmse_cie))

#         # compare differnet between the obtained CIE and the actual target CIE in the original space
#         x_pred_raw = x_pred *range_[:x_dim] + min_[:x_dim]
#         x_raw =  x *range_[:x_dim] +min_[:x_dim]        
#         rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
#         if show==1:
#             print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

#     return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()

def evaluate_forward_minmax_dataset(forward_model, dataset, show=0):
    # for evaluate the dataset itself.
    # x: structure ; y: CIE 
    # return: predicted CIE
    forward_model.eval()
    with torch.no_grad():
        range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]
        M = x.size()[0]
        y_pred = forward_model(x, None)
        y_pred_raw = y_pred *range_[x_dim:] + min_[x_dim:]
        y_raw = y *range_[x_dim:] + min_[x_dim:]

        # get MSE for the design
        rmse_cie = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        rmse_cie_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
        
        if show==1:
            print('Reconstruct net RMSE loss {:.3f}'.format(rmse_cie))
            print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))
        
    y_pred_raw = y_pred_raw.cpu().numpy()
    y_raw = y_raw.cpu().numpy()
    return   y_raw, y_pred_raw

def evaluate_forward_minmax(forward_model, dataset, param_pred):
    # for evaluate any data, the param_pred is not normalized
    # x: structure ; y: CIE 
    # return: predicted CIE
    
    forward_model.eval()
    
    with torch.no_grad(): 
        range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]
        M = np.shape(param_pred)[0]
        param_cie = np.zeros([M, 3])
        param = np.concatenate((param_pred, param_cie), axis=1)
        x_pred = dataset.scaler.transform(param)[:, :x_dim]

        y_pred = forward_model(torch.tensor(x_pred).float().to(DEVICE), None)
        y_pred_raw = y_pred *range_[x_dim:] + min_[x_dim:]
        
    y_pred_raw = y_pred_raw.cpu().numpy()
    return  y_pred_raw


def evaluate_tandem_minmax_accuracy(model, forward_model, dataset, show=1):
    # evaluate the tandem prediction using any desired forward model
    # x: structure ; y: CIE 
    
    '''
    returns:
        y_raw: original desired xyY
        y_raw_pred: xyY predicted by the forward module for the inversely designed structure
        x_raw: original structure parameters
        x_raw_pred: inversely designed parameters.
    '''

    model.eval()
    forward_model.eval()
    with torch.no_grad():
        range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        x_pred = model.inverse_model(y, None)
        y_pred = forward_model(x_pred, None)

        # get original data
        x_pred_raw = x_pred *range_[:x_dim] + min_[:x_dim]
        x_raw =  x *range_[:x_dim] +min_[:x_dim] 
        
        y_pred_raw = y_pred *range_[x_dim:] +min_[x_dim:]
        y_raw =  y *range_[x_dim:] +min_[x_dim:]
        
        # get MSE for the design
        rmse_design = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        rmse_design_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
        rmse_cie = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        rmse_cie_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
        
        if show==1:
            print("Tandem net Design RMSE loss {:.3f}".format(rmse_design.item()))
            print('Tandem Design RMSE raw loss {:.3f}'.format(rmse_design_raw.item()))
            print('Reconstruct net RMSE loss {:.3f}'.format(rmse_cie))
            print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))
            

    return y_raw.cpu().numpy(), x_raw.cpu().numpy(), y_pred_raw.cpu().numpy(), x_pred_raw.cpu().numpy()

def evaluate_vae_GSNN_minmax_inverse(vae_GSNN_model, forward_model, dataset, show=1):
    # evaluate both the vae_GSNN and vae_hybrid model using a forward model
    # x: structure. y: CIE
    '''
    returns:
        y_raw: original desired xyY
        y_raw_pred: xyY predicted by the forward module for the inversely designed structure
        x_raw: original structure parameters
        x_raw_pred: inversely designed parameters.
    '''
    vae_GSNN_model.eval()
    forward_model.eval()
    
    with torch.no_grad():
        range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        # inferenc using vae_GSNN model and predict using forward model
        x_pred, mu, logvar, temp = vae_GSNN_model.inference(y)
        y_pred = forward_model(x_pred, None)

        x_pred_raw = x_pred *range_[:x_dim] + min_[:x_dim]
        x_raw =  x *range_[:x_dim] +min_[:x_dim] 
        
        y_pred_raw = y_pred *range_[x_dim:] +min_[x_dim:]
        y_raw =  y *range_[x_dim:] +min_[x_dim:]

        # get MSE for the design
        rmse_design = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        rmse_design_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
        rmse_cie = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        rmse_cie_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())

        if show==1:
            print("VAE net Design RMSE loss {:.3f}".format(rmse_design.item()))
            print('VAE Design RMSE raw loss {:.3f}'.format(rmse_design_raw.item()))
            print('Reconstruct net RMSE loss {:.3f}'.format(rmse_cie))
            print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return y_raw.cpu().numpy(), x_raw.cpu().numpy(), y_pred_raw.cpu().numpy(), x_pred_raw.cpu().numpy()

def evaluate_gan_minmax_inverse(gan_model, forward_model, dataset, show=1):
    # evaluate both the gan model using a forward model
    # y: structure. x: CIE

    '''
    returns:
        x_raw: original desired xyY
        x_raw_pred: xyY predicted by the forward module for the inversely designed structure
        y_raw: original structure parameters
        y_raw_pred: inversely designed parameters.
    '''

    gan_model.eval()
    with torch.no_grad():
        range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        z = gan_model.sample_noise(len(x)).to(DEVICE)
        y_pred = gan_model.generator(x, z)
        x_pred = forward_model(y_pred, None)

        x_pred_raw = x_pred *range_[:x_dim] + min_[:x_dim]
        x_raw =  x *range_[:x_dim] +min_[:x_dim] 
        
        y_pred_raw = y_pred *range_[x_dim:] +min_[x_dim:]
        y_raw =  y *range_[x_dim:] +min_[x_dim:]

        # get MSE for the design
        rmse_design = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
        rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())

        if show==1:
            print("GAN net Design RMSE loss {:.3f}".format(rmse_design.item()))
            print('GAN Design RMSE raw loss {:.3f}'.format(rmse_design_raw.item()))
            print('Reconstruct net RMSE loss {:.3f}'.format(rmse_cie))
            print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()


def evaluate_inn_minmax_inverse(model, forward_model, dataset, show = 1):
    # x: structure. y: CIE

    model.eval()
    
    range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)

    def infer_design(model, dataset):
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]
        y_clean = y.clone()

        batch_size = len(x)
        pad_x, pad_yz = model.create_padding(batch_size)
        pad_x = pad_x.to(DEVICE)
        pad_yz = pad_yz.to(DEVICE)

        y += model.y_noise_scale * torch.randn(batch_size, y.size(1)).float().to(DEVICE)
        y = torch.cat((torch.randn(batch_size, model.dim_z).float().to(DEVICE), pad_yz, y), dim = 1)

        y = y_clean + model.y_noise_scale * torch.randn(batch_size, model.dim_y).to(DEVICE)

        y_rev_rand = torch.cat((torch.randn(batch_size, model.dim_z).to(DEVICE), pad_yz, y), dim=1)
        output_rev_rand = model(y_rev_rand, rev=True)[0]
        x_pred = output_rev_rand[:, :model.dim_x]

        return x_pred

    with torch.no_grad():
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        x_pred = infer_design(model, dataset)
        y_pred = forward_model(x_pred, y)
    
        x_pred_raw = x_pred *range_[:x_dim] + min_[:x_dim]
        x_raw =  x *range_[:x_dim] +min_[:x_dim] 
        
        y_pred_raw = y_pred *range_[x_dim:] +min_[x_dim:]
        y_raw =  y *range_[x_dim:] +min_[x_dim:]

        # get MSE for the design
        rmse_design = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        rmse_design_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
        rmse_cie = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        rmse_cie_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())

        if show==1:
            print("INN net Design RMSE loss {:.3f}".format(rmse_design.item()))
            print('INN Design RMSE raw loss {:.3f}'.format(rmse_design_raw.item()))
            print('Reconstruct net RMSE loss {:.3f}'.format(rmse_cie))
            print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return y_raw.cpu().numpy(), x_raw.cpu().numpy(), y_pred_raw.cpu().numpy(), x_pred_raw.cpu().numpy()




def evaluate_gan_inverse(forward_model, gan_model, configs, dataset):

    gan_model.eval()
    with torch.no_grad():
        mean, std = torch.tensor(dataset.scaler.mean_).to(DEVICE), torch.tensor(np.sqrt(dataset.scaler.var_)).to(DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        z = gan_model.sample_noise(len(x)).to(DEVICE)
        y_pred = gan_model.generator(x, z)

        # get MSE for the design
        rmse = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        print("Simple net Design RMSE loss {:.3f}".format(rmse.item()))

        # get RMSE
        y_pred_raw = y_pred * std[x_dim:] + mean[x_dim:]
        y_raw= y * std[x_dim:] + mean[x_dim:]
        rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
        print('Simple net RMSE loss {:.3f}'.format(rmse_design_raw.item()))

        # get difference between the obtained CIE and the actual target CIE
        x_pred = forward_model(y_pred, y)
        rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss {:.3f}'.format(rmse_cie))

        # compare differnet between the obtained CIE and the actual target CIE in the original space
        x_pred_raw = x_pred * std[:x_dim] + mean[:x_dim]
        x_raw = x * std[:x_dim] + mean[:x_dim]
        rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
        print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()


def evaluate_inn_inverse(forward_model, model, configs, dataset, show = 1):
    '''
    The dataset need to be in inverse format, i.e., x corresponds to target while y corresponds to design.
    '''

    model.eval()

    mean, std = torch.tensor(dataset.scaler.mean_).to(DEVICE), torch.tensor(np.sqrt(dataset.scaler.var_)).to(DEVICE)

    def infer_design(model, dataset):
        x, y = dataset.y.to(DEVICE), dataset.x.to(DEVICE)
        x_dim = x.size()[1]
        y_clean = y.clone()

        batch_size = len(x)
        pad_x, pad_yz = model.create_padding(batch_size)
        pad_x = pad_x.to(DEVICE)
        pad_yz = pad_yz.to(DEVICE)

        y += model.y_noise_scale * torch.randn(batch_size, y.size(1)).float().to(DEVICE)
        y = torch.cat((torch.randn(batch_size, model.dim_z).float().to(DEVICE), pad_yz, y), dim = 1)

        y = y_clean + model.y_noise_scale * torch.randn(batch_size, model.dim_y).to(DEVICE)

        y_rev_rand = torch.cat((torch.randn(batch_size, model.dim_z).to(DEVICE), pad_yz, y), dim=1)
        output_rev_rand = model(y_rev_rand, rev=True)
        x_pred = output_rev_rand[:, :model.dim_x]

        return x_pred
    
    with torch.no_grad():
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        y_pred = infer_design(model, dataset)

        # get MSE for the design
        rmse = torch.sqrt((y_pred - y).pow(2).sum(dim=1).mean())
        if show==1:
            print("Simple net Design RMSE loss {:.3f}".format(rmse.item()))

        # get RMSE
        y_pred_raw = y_pred * std[x_dim:] + mean[x_dim:]
        y_raw= y * std[x_dim:] + mean[x_dim:]
        rmse_design_raw = torch.sqrt((y_pred_raw - y_raw).pow(2).sum(dim=1).mean())
        if show==1:
            print('Simple net RMSE loss {:.3f}'.format(rmse_design_raw.item()))

        # get difference between the obtained CIE and the actual target CIE
        x_pred = forward_model(y_pred, y)
        rmse_cie = torch.sqrt((x_pred - x).pow(2).sum(dim=1).mean())
        if show==1:
            print('Reconstruct RMSE loss {:.3f}'.format(rmse_cie))

        # compare differnet between the obtained CIE and the actual target CIE in the original space
        x_pred_raw = x_pred * std[:x_dim] + mean[:x_dim]
        x_raw = x * std[:x_dim] + mean[:x_dim]
        rmse_cie_raw = torch.sqrt((x_pred_raw - x_raw).pow(2).sum(dim=1).mean())
        if show==1:
            print('Reconstruct RMSE loss raw {:.3f}'.format(rmse_cie_raw))

    return x_raw.cpu().numpy(), y_raw.cpu().numpy(), x_pred_raw.cpu().numpy(), y_pred_raw.cpu().numpy()

# def evaluate_forward(forward_model, dataset, param_pred):
#     '''
#     The dataset need to be in inverse format, i.e., x corresponds to target while y corresponds to design.
#     '''
#     forward_model.eval()
    
#     N = np.shape(param_pred)[0]
#     param_pred = np.reshape(param_pred, (-1, 4))
    
#     with torch.no_grad():
#         mean, std = torch.tensor(dataset.scaler.mean_).to(DEVICE), torch.tensor(np.sqrt(dataset.scaler.var_)).to(DEVICE)
#         x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
#         x_dim = x.size()[1]
#         M = np.shape(param_pred)[0]
#         param_cie = np.zeros([M, 3])
#         param_y_cie = np.concatenate((param_cie, param_pred), axis=1)
#         y_pred = dataset.scaler.transform(param_y_cie)[:, 3:7]

#         x_pred = forward_model(torch.tensor(y_pred).float(), y)
#         x_pred_raw = x_pred * std[:x_dim] + mean[:x_dim]
        
#     x_pred_raw = x_pred_raw.cpu().numpy()
#     return  np.reshape(x_pred_raw, (N, -1))

# def evaluate_minmax_forward_dataset(forward_model, dataset):
#     # for evaluate the dataset itself.
#     '''
#     The dataset need to be in inverse format, i.e., x corresponds to target while y corresponds to design.
#     '''
#     forward_model.eval()
#     #print(list(forward_model.parameters()))
#     with torch.no_grad():
#         range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)
#         x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
#         x_dim = x.size()[1]
#         M = x.size()[0]
#         x_pred = forward_model.forward(y, None)
#         #x_pred_raw = x_pred *range_[:x_dim] + min_[:x_dim]
#         #x_raw = x *range_[:x_dim] + min_[:x_dim]
        
#     #x_pred_raw = x_pred_raw.cpu().numpy()
#     #x_raw = x_raw.cpu().numpy()
#     x_pred_raw = x_pred.cpu().numpy()
#     x_raw = x.cpu().numpy()
#     return  x_pred_raw, x_raw

# def evaluate_minmax_forward(forward_model, dataset, param_pred):
#     '''
#     The dataset need to be in inverse format, i.e., x corresponds to target while y corresponds to design.
#     '''
#     forward_model.eval()
#     N = np.shape(param_pred)[0]
    
#     with torch.no_grad(): 
#         range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)
#         x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
#         x_dim = x.size()[1]
#         M = np.shape(param_pred)[0]
#         param_cie = np.zeros([M, 3])
#         param_y_cie = np.concatenate((param_cie, param_pred), axis=1)
#         y_pred = dataset.scaler.transform(param_y_cie)[:, 3:7]
#         #y_pred = (param_pred - min_[x_dim:])/range_[x_dim:]

#         x_pred = forward_model(torch.tensor(y_pred).float(), y)
#         x_pred_raw = x_pred *range_[:x_dim] + min_[:x_dim]
        
#     x_pred_raw = x_pred_raw.cpu().numpy()
#     return  x_pred_raw


def count_params(model):

    return sum([np.prod(layer.size()) for layer in model.parameters() if layer.requires_grad])

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# multiscale MMD loss
def MMD_multiscale(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz

    XX, YY, XY = (torch.zeros(xx.shape).to(DEVICE),
                  torch.zeros(xx.shape).to(DEVICE),
                  torch.zeros(xx.shape).to(DEVICE))

    for a in [0.05, 0.2, 0.9]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    return torch.mean(XX + YY - 2.*XY)


def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()

def MMD_loss(latent, LATENT_SIZE =3, batch_size=128):

    return MMD(torch.randn(batch_size, LATENT_SIZE, requires_grad = False).to(DEVICE), latent)
