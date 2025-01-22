import torch

import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from .vae_basics import DenseNet, sampling, elbo, elbo_w_err, elbo_student_t
from .helper import try_gpu

from tqdm import tqdm
import pickle


class VAE(nn.Module):
    
    '''
    Initialize a VAE model with assigned parameters

    Parameters:
    -----------
    n_dim_i : int
        Input dimensionality

    n_dim_o : int
        Output dimensionality
    
    n_dim_latent : int, default 1
        Number of dimensionality in latent space

    n_hidden_layers : int, or [int, int]
        Number of hidden layers in the encoder and decoder. If an int is given, it will applied to both encoder and decoder;
        If a length 2 list is given, first int will be used for encoder, the second will be used for decoder

    n_hidden_size : int, or [int, int], or array of int
        Number of units in hidden layers. If an int is given, i will be applied to all hidden layers in both encoder and decoder;
        If a length 2 array is given, first int will be used for all layers in the encoder, the second will be used for the decoder.
        Or an array with length equal to the number of hidden layers can be given, the number of units will be assigned accordingly.

    activation : str, default tanh
        activation function for the hidden layers
    '''
    
    def __init__(self, n_dim_i, n_dim_o, n_dim_latent=1, n_hidden_layers=3, n_hidden_size=100, activation=torch.tanh, device=try_gpu()):     
        
        super(VAE, self).__init__()
        
        if type(n_hidden_layers) is int:
            self.n_layer_encoder = n_hidden_layers
            self.n_layer_decoder = n_hidden_layers
        elif len(n_hidden_layers) == 2:
            self.n_layer_encoder = n_hidden_layers[0]
            self.n_layer_decoder = n_hidden_layers[1]
        else:
            raise ValueError("Please provide legal n_hidden_layers!")
        
        # List of hidden units in encoder and decoder
        if type(n_hidden_size) is int:
            self.n_size_encoder = [n_hidden_size]*self.n_layer_encoder
            self.n_size_decoder = [n_hidden_size]*self.n_layer_decoder
        elif len(n_hidden_size) == 2:
            self.n_size_encoder = [n_hidden_size[0]]*self.n_layer_encoder
            self.n_size_decoder = [n_hidden_size[1]]*self.n_layer_decoder
        elif len(n_hidden_size) == self.n_layer_encoder+self.n_layer_decoder:
            self.n_size_encoder = n_hidden_size[:self.n_layer_encoder]
            self.n_size_decoder = n_hidden_size[self.n_layer_encoder:]
        else: 
            raise ValueError("Please provide legal n_hidden_size!")
        
        self.dim_x = n_dim_i
        self.dim_y = n_dim_o
        self.activation = activation
        self.dim_z = n_dim_latent
        self.device = device
        
        self.encoder = DenseNet(self.dim_x, self.dim_z * 2, self.n_layer_encoder+1, self.n_size_encoder, self.activation).to(device)
        self.decoder = DenseNet(self.dim_z, self.dim_y, self.n_layer_decoder+1, self.n_size_decoder, self.activation).to(device)
        
        self.loss_train = []
        self.loss_names = ["Loss", "NLL", "KL_div"]
        
    def sample(self, n_sample=1000, mu=0, sigma=1):
        z = mu + sigma * torch.randn(n_sample, self.dim_z, device=self.device)
        x = self.decoder(z)
        return x

    def reconstruct(self, input_x,ml_recon=False, repeats=1):
        '''
        Reconstructs output from the VAE

        Parameters :
        ------------
        input_x (): input for the VAE
        ml_recon (bool): Flag which determined whether to return the most likely reconstruction (if True),
        or a randomly sampled one (default: False)
        repeats (int) : number of times to sample output from the VAE (default 1) 

        Returns :
        ---------
        recons (array or list of arrays): output from the decoder.
        
        '''
        encoding = self.encoder(input_x.to(self.device))
        z_mean, z_log_var = encoding[:, :self.dim_z], encoding[:, self.dim_z:]
        
        if ml_recon:
            repeats=1
        if repeats==1:
            if ml_recon==False:
                z = sampling(z_mean, z_log_var)
                recons = self.decoder(z)
            else:
                recons = self.decoder(z_mean)
        else:
            recons=[]
            for k in range(repeats):
                z = sampling(z_mean, z_log_var)
                recons.append(self.decoder(z))
                
        return recons
        
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            D = pickle.load(f)
        vae = cls(D['n_dim_i'], D['n_dim_o'], D['n_dim_latent'], D['n_hidden_layers'], D['n_hidden_size'], D['activation'])
        vae.load_state_dict(D['state_dict'])
        vae.loss_train = D['loss_train']
        return vae

    def save(self, filepath):
        '''
        Customized save function using pickle.
        '''
        D = {}
        D['n_dim_i'] = self.dim_x
        D['n_dim_o'] = self.dim_y
        D['n_dim_latent'] = self.dim_z
        D['n_hidden_layers'] = [self.n_layer_encoder, self.n_layer_decoder]
        D['n_hidden_size'] = self.n_size_encoder + self.n_size_decoder
        D['activation'] = self.activation
        D['loss_train'] = self.loss_train
        D['state_dict'] = self.state_dict()

        with open(filepath, 'wb') as f:
            pickle.dump(D, f, pickle.HIGHEST_PROTOCOL)

    
    def train(self, x_train, y_train, e_train, optim, x_val=None, y_val=None, e_val=None, epochs=10, batch_size=256, w_kl=1.0,eps=0.05, include_errors=False, stdof=None, verbose=True):

        if isinstance(batch_size, int):
            batch_size = [batch_size] * epochs
        elif isinstance(batch_size, list):
            assert len(batch_size) == epochs
        
        if x_val is not None and y_val is not None:
            if include_errors:
                dataset_val = TensorDataset(x_val, y_val,e_val)
            else:
                dataset_val = TensorDataset(x_val, y_val)
            sampler = RandomSampler(dataset_val)
            valloader = DataLoader(dataset_val, batch_size=256, sampler=sampler)
        
        dataset_train = TensorDataset(x_train, y_train, e_train)
        for epoch in range(epochs):
            trainloader = DataLoader(dataset_train, batch_size=batch_size[epoch], shuffle=True)
            progress_bar = tqdm(trainloader, desc=f"Ep {epoch+1}")

            if include_errors:
                for x_batch, y_batch, e_batch in progress_bar:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    e_batch = e_batch.to(self.device)
                    optim.zero_grad()
                    encoding = self.encoder(x_batch)
                    z_mean, z_log_var = encoding[:, :self.dim_z], encoding[:, self.dim_z:]
                    z = sampling(z_mean, z_log_var)
                    recons_x = self.decoder(z)

                    if stdof is None:
                        loss_train, nll_train, kl_train = elbo_w_err(    z_mean, z_log_var, y_batch, recons_x, e_batch, w_kl, eps=eps)                        
                    else:
                        loss_train, nll_train, kl_train = elbo_student_t(z_mean, z_log_var, y_batch, recons_x, e_batch, w_kl, eps=eps, stdof=stdof)
                        
                    loss_train.backward()
                    optim.step()
                    
                    if x_val is not None and y_val is not None:
                        x_batch_test, y_batch_test, e_batch_test = next(iter(valloader))
                        x_batch_test = x_batch_test.to(self.device)
                        y_batch_test = y_batch_test.to(self.device)
                        e_batch_test = e_batch_test.to(self.device)
                        # print("e_batch_test: " + str(e_batch_test.size()))
                        
                        encoding_test = self.encoder(x_batch_test)
                        z_mean_test, z_log_var_test = encoding_test[:, :self.dim_z], encoding_test[:, self.dim_z:]
                        z_test = sampling(z_mean_test, z_log_var_test)
                        recons_x_test = self.decoder(z_test)

                        if stdof is None:
                            loss_test, nll_test, kl_test = elbo_w_err(z_mean_test, z_log_var_test, y_batch_test, recons_x_test, e_batch_test, w_kl,eps=eps, verbose=verbose)
                        else:
                            loss_test, nll_test, kl_test = elbo_student_t(z_mean_test, z_log_var_test, y_batch_test, recons_x_test, e_batch_test, w_kl,eps=eps, stdof=stdof, verbose=verbose)
    
                        loss_np, loss_test_np = loss_train.item(), loss_test.item()
                        # abbreviated to fit display...
                        progress_bar.set_postfix(Train=loss_np, Test=loss_test_np, mem=torch.cuda.memory_allocated()/1e9)
                        self.loss_train.append([loss_np, nll_train.item(), kl_train.item(), loss_test_np, nll_test.item(), kl_test.item()])  
                    else:
                        loss_np = loss_train.item()
                        progress_bar.set_postfix(Trainloss=loss_np, memory=torch.cuda.memory_allocated()/1e9)
                        self.loss_train.append([loss_np, nll_train.item(), kl_train.item()])
            else: 
                for x_batch, y_batch, _ in progress_bar:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    optim.zero_grad()
                    encoding = self.encoder(x_batch)
                    z_mean, z_log_var = encoding[:, :self.dim_z], encoding[:, self.dim_z:]
                    z = sampling(z_mean, z_log_var)
                    recons_x = self.decoder(z)
                    loss_train, nll_train, kl_train = elbo(z_mean, z_log_var, y_batch, recons_x, w_kl)
                    loss_train.backward()
                    optim.step()
                    
                    if x_val is not None and y_val is not None:
                        x_batch_test, y_batch_test = next(iter(valloader))
                        x_batch_test = x_batch_test.to(self.device)
                        y_batch_test = y_batch_test.to(self.device)
                        
                        encoding_test = self.encoder(x_batch_test)
                        z_mean_test, z_log_var_test = encoding_test[:, :self.dim_z], encoding_test[:, self.dim_z:]
                        z_test = sampling(z_mean_test, z_log_var_test)
                        recons_x_test = self.decoder(z_test)

                        loss_test, nll_test, kl_test = elbo(z_mean_test, z_log_var_test, y_batch_test, recons_x_test, w_kl)
                        loss_np, loss_test_np = loss_train.item(), loss_test.item()
                        progress_bar.set_postfix(Train=loss_np, Test=loss_test_np, mem=torch.cuda.memory_allocated()/1e9)
                        self.loss_train.append([loss_np, nll_train.item(), kl_train.item(), loss_test_np, nll_test.item(), kl_test.item()])  
                    else:
                        loss_np = loss_train.item()
                        progress_bar.set_postfix(Trainloss=loss_np, memory=torch.cuda.memory_allocated()/1e9)
                        self.loss_train.append([loss_np, nll_train.item(), kl_train.item()])