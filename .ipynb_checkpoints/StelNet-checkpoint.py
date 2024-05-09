# IMPORT LIBRARIES

import random as rn
import numpy as np
import math as mt

import matplotlib.pyplot as plt
from matplotlib import cm

import pylab as pl
import sys
from matplotlib.pyplot import cm
import glob
import pandas as pd

from astropy.table import Table
import torch
import torch.nn as nn


# Normalize Data

def normalize(data, photometry=False):
    """
    if photometry==True, first 3 columns of data should be 3 gaia photometry bands for each object
    if photometry==False, first 2 columns of data should be logTeff, logLum for each object
    """
    if photometry==True:
        norm_min = np.load("Aux/norm_min_payne.npy", allow_pickle=True)
        norm_max = np.load("Aux/norm_max_payne.npy", allow_pickle=True)

        x_data = torch.from_numpy((data[:,:3] - norm_min[:3]) / (norm_max[:3]-norm_min[:3]))
        return x_data
        
    if photometry==False:
        norm_min_preMS = np.load('Aux/norm_min_preMS.npy')
        norm_max_preMS = np.load('Aux/norm_max_preMS.npy')
        norm_min_postMS = np.load('Aux/norm_min_postMS.npy')
        norm_max_postMS = np.load('Aux/norm_max_postMS.npy')
        x_data_preMS = torch.from_numpy((data[:,:2]-norm_min_preMS[:2])/(norm_max_preMS[:2]-norm_min_preMS[:2])).float()
        x_data_postMS = torch.from_numpy((data[:,:2]-norm_min_postMS[:2])/(norm_max_postMS[:2]-norm_min_postMS[:2])).float()

        return x_data_preMS, x_data_postMS


def unnormalize_teff_lum(y):
    """
    y comes from the neural network prediction
    """
    norm_min = np.load("Aux/norm_min_payne.npy", allow_pickle=True)
    norm_max = np.load("Aux/norm_max_payne.npy", allow_pickle=True)
    y_un = y*(np.array(norm_max_payne[3:]-norm_min[3:])) + norm_min[3:]
    return y_un


def unnormalize_age_mass(y_pre, y_post):
    # ADD OPTION TO SpECIFY WHETHER WE ARE UN/NORMALIZING PHOTOMETRY OR TEFF?LUM OR AGE?MASS
    norm_min_preMS = np.load('Aux/norm_min_preMS.npy')
    norm_max_preMS = np.load('Aux/norm_max_preMS.npy')
    norm_min_postMS = np.load('Aux/norm_min_postMS.npy')
    norm_max_postMS = np.load('Aux/norm_max_postMS.npy')
    y_pre_un = y_pre*(np.array(norm_max_preMS[2:])-np.array(norm_min_preMS[2:]))+ np.array(norm_min_preMS[2:])
    y_post_un = y_post*(np.array(norm_max_postMS[2:])-np.array(norm_min_postMS[2:]))+ np.array(norm_min_postMS[2:])

    return y_pre_un, y_post_un


# Build Model

class NN(nn.Module):
    def __init__(self, D_in, D_out, num_layers, num_nodes, activation):
        super(NN, self).__init__()
        
        # Specify list of layer sizes 
        sizes = [D_in] + [num_nodes] * num_layers + [D_out]
        in_sizes, out_sizes = sizes[:-1], sizes[1:]
        
        # Construct linear layers
        self.linears = nn.ModuleList()
        for n_in, n_out in zip(in_sizes, out_sizes):
            self.linears.append(nn.Linear(n_in, n_out))
        
        # Specify activation function 
        self.activation = activation
        
    def forward(self, x):
        
        for l in self.linears[:-1]:
            x = self.activation(l(x))
        x = self.linears[-1](x)
        
        return x


# Predict
## prediction for photometry
def predict_surface_params(X):
    x_data = normalize(X, photometry=True)
    
    D_in=3
    D_out=2
    num_layers=10 # change if photometry NN architeture changes
    num_nodes=10 # ^^
    activation=nn.ReLU()
    net=NN(D_in, D_out, num_layers, num_nodes, activation)
    modelname = "Models/Photometry/payne_10layer_10node_5000it_1em3lr_full_physical_dataset_NOSCHEDULER_osc"# this is an example model with the example hyperparameters trained on the full payne dataset
    net.load_state_dict(torch.load(modelname), strict=False)
    y_pred = torch.unsqueeze(net(x_data),0).detatch().numpy()[0]
    y_pred_un = unnormalize_teff_lum(y_pred)
    return y_pred_un

## prediction for teff, lum->age, mass
def predict_age_mass(X, n=20, TL=None):
    x_data_preMS, x_data_postMS = normalize(X, photometry=False) # specify normalizing a teff and lum
    
    D_in = 2
    D_out = 2
    num_layers = 10
    num_nodes =50
    activation = nn.ReLU()    
    net = NN(D_in, D_out, num_layers, num_nodes, activation)
    net_preMS = NN(D_in, D_out, num_layers, num_nodes, activation)
    net_postMS = NN(D_in, D_out, num_layers, num_nodes, activation)
    num_models=n
    if num_models > 20: sys.exit('Number of models should not exceed 20')

    # Baseline model
    pre_model = 'Models/Baseline/mist_baseline_preMS{}'
    post_model = 'Models/Baseline/mist_baseline_postMS{}'


    # Transfer Learning Options
    if TL=='DH':
        # Transfer Learning with D&H from Garraffo+2021
        pre_model = 'Models/TL/DH/mist_DH_preMS{}'
        post_model = 'Models/TL/DH/mist_DH_postMS{}'

    if TL=="reu2023":
        # Transfer Learning from SAO REU 2023
        pre_model = "Models/TL/reu2023_preZAMS_TL/ONC_NGC6530_xtra_lowhighmassMIST{}"
        post_model = "Models/TL/reu2023_postZAMS_TL/FGK_DH_GR_xtra_lowhighmassMIST{}" 


    ## add more TL options here for models made using transfer_learning_bootstraps()
    ### example:
    if TL=="example":
        # example for github
        # not altering pre_model for now
        post_model = "Models/TL/new_TL/github_example/postms_tl_example{}"
    
    net_preMS.load_state_dict(torch.load(pre_model.format(0)), strict=False)
    net_postMS.load_state_dict(torch.load(post_model.format(0)), strict=False)
    y_pred_preMS = torch.unsqueeze(net_preMS(x_data_preMS),0).detach().numpy()
    y_pred_postMS = torch.unsqueeze(net_postMS(x_data_postMS),0).detach().numpy()

    for i in range(1,n):
        net_preMS.load_state_dict(torch.load(pre_model.format(i)), strict=False)
        net_postMS.load_state_dict(torch.load(post_model.format(i)), strict=False)
        y_pred_preMS = np.append(y_pred_preMS, torch.unsqueeze(net_preMS(x_data_preMS),0).detach().numpy(), axis=0)
        y_pred_postMS = np.append(y_pred_postMS, torch.unsqueeze(net_postMS(x_data_postMS),0).detach().numpy(), axis=0)
    
    y_pred_preMS_un, y_pred_postMS_un = unnormalize_age_mass(y_pred_preMS, y_pred_postMS)

    return y_pred_preMS_un, y_pred_postMS_un


# Posterior Statistics for Each Model

def stats(y_pred):
    y_mean = np.mean(y_pred, 0)
    y_std = np.std(y_pred, 0)

    return y_mean, y_std


# Mixture of Models

def pis(y_mean_preMS, y_mean_postMS):
    boundary = pd.read_pickle('Aux/boundary')
    edge = pd.read_pickle('Aux/edge')
    
    mass_pred_pre = 10**(y_mean_preMS[:,1])
    mass_pred_post = 10**(y_mean_postMS[:,1])
    idx_bd_pre = np.argmin(abs(mass_pred_pre[:,np.newaxis]-np.array(boundary['star_mass'])),axis=1)
    idx_bd_post = np.argmin(abs(mass_pred_post[:,np.newaxis]-np.array(boundary['star_mass'])),axis=1)
    idx_ed_pre = np.argmin(abs(mass_pred_pre[:,np.newaxis]-np.array(edge['star_mass'])),axis=1)
    idx_ed_post = np.argmin(abs(mass_pred_post[:,np.newaxis]-np.array(edge['star_mass'])),axis=1)
    
    m=np.array([0.02,0.05,0.1,0.4, 0.6,0.8])
    Chi_pre= np.clip(((mass_pred_pre-0.08)/(0.42)+1).astype(int)+0.3, 0.3, 2.3)
    Chi_post = np.clip(((mass_pred_post-0.08)/(0.42)+1).astype(int)+0.3, 0.3, 2.3)
    w_pre = mass_pred_pre**(-Chi_pre) * np.array(boundary['star_age'].iloc[idx_bd_pre])
    w_post = mass_pred_post**(-Chi_post) * (np.array(edge['star_age'].iloc[idx_ed_post])-np.array(boundary['star_age'].iloc[idx_bd_post]))
    w_tot = w_pre + w_post            
    w_pre_norm = w_pre/w_tot
    w_post_norm = w_post/w_tot

    return w_pre_norm, w_post_norm


def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))/(sig*mt.sqrt(2*mt.pi))

    
def Gaussian_posteriors(y_mean_preMS, y_mean_postMS, y_std_preMS, y_std_postMS, pi_pre, pi_post, n_obs, num_x_points=500):
    x_values=np.linspace(0, 1, num_x_points)
    x= (x_values[:,np.newaxis]*np.ones(n_obs))[:,:,np.newaxis]*np.ones(2)*[9,6]+[4,-2]
    pre=(np.ones(num_x_points)[:,np.newaxis]*pi_pre[np.newaxis,:])[:,:,np.newaxis]*np.ones(2)
    post=(np.ones(num_x_points)[:,np.newaxis]*pi_post[np.newaxis,:])[:,:,np.newaxis]*np.ones(2)
    y_gaussian_posteriors = pre* gaussian(x, y_mean_preMS, y_std_preMS)+ post* gaussian(x, y_mean_postMS, y_std_postMS)
    return y_gaussian_posteriors


# Posterior Probability Distributions

def posteriors(y_pre , y_post, pi_pre, pi_post, n =20):
    y_posteriors = (np.ones((n, 1))*pi_pre[np.newaxis])[:,:,np.newaxis]*np.ones(2) * y_pre+ (np.ones((n, 1))*pi_post[np.newaxis])[:,:,np.newaxis]*np.ones(2) * y_post 
    return y_posteriors


# Plot

def plot_multiple_posteriors(y_posteriors, obs_array, n=20, dotsize = 2):
    fig, ax = plt.subplots(2, 1, figsize=(20, 15), sharex= True)
    for i in obs_array:
        ax[0].scatter(np.ones(n)*i, y_posteriors[:,i,0], s=dotsize)
        ax[1].scatter(np.ones(n)*i, y_posteriors[:,i,1], s=dotsize)
    ax[1].set_xlabel('Observation id',fontsize=30)
    ax[0].set_ylabel('$\log(age \ [yrs])$', fontsize=30)
    ax[1].set_ylabel('$\log(mass)$ [$M_{\odot}$]', fontsize=30)
    ax[0].tick_params(labelsize=25)
    ax[1].tick_params(labelsize=25)

    return ax

def plot_posterior(y_post,obs_id, n=20):
    fig, ax = plt.subplots(2, 1, figsize=(20, 15), sharex= True)
    ax[0].scatter(np.arange(n)+1, y_post[:,obs_id,0], s=30)
    ax[1].scatter(np.arange(n)+1, y_post[:,obs_id,1], s=30)
    ax[0].set_xticks([i for i in range(1,21)])
    ax[0].tick_params(labelsize=25)
    ax[1].tick_params(labelsize=25)
    ax[1].set_xlabel('model number', fontsize=30)
    ax[0].set_ylabel('$\log(age \ [yrs])$', fontsize=30)
    ax[1].set_ylabel('$\log(mass \ [M_{\odot}])$', fontsize=30)
    
    return ax

def plot_posterior_hist(y_post, obs_id, bins=15, log_age_range=[5,12], log_mass_range=[-1.5,1.5]):
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(30,10), sharex = False)
    ax[0].hist(y_post[:,obs_id,0], bins=15)
    ax[1].hist(y_post[:,obs_id,1], bins=15)   
    ax[0].tick_params(labelsize=25)
    ax[1].tick_params(labelsize=25)
    ax[0].set_yscale('log')
    ax[0].set_xlabel('$\log(age \ [yrs])$', fontsize=30)
    ax[1].set_xlabel('$\log(mass \ [M_{\odot}])$', fontsize=30)
    ax[0].set_ylabel('number of predictions', fontsize=30)
    
    return ax

def plot_gaussian_posteriors(y_gaussian_posteriors, obs_id, log_age_range=[4,12], log_mass_range=[-1.5,1.5]):
    x_age = np.linspace(log_age_range[0], log_age_range[1], y_gaussian_posteriors.shape[0])
    x_mass = np.linspace(log_mass_range[0], log_mass_range[1], y_gaussian_posteriors.shape[0])
    y = y_gaussian_posteriors[:,obs_id,:]
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(30,10), sharex = False)
    ax[0].plot(x_age,0.01+y[:,0])
    ax[1].plot(x_mass,0.01+y[:,1], 'm')     
    ax[0].set_xlim(log_age_range[0],log_age_range[1])
    ax[1].set_xlim(log_mass_range[0],log_mass_range[1])
    ax[0].tick_params(labelsize=25)
    ax[1].tick_params(labelsize=25)
    ax[0].set_yscale('log')
    ax[0].set_xlabel('$\log(age \ [yrs])$', fontsize=30)
    ax[1].set_xlabel('$\log(mass \ [M_{\odot}])$', fontsize=30)
    ax[0].set_ylabel('Probability', fontsize=30)
    ax[1].text(log_mass_range[1]-(log_mass_range[1]-log_mass_range[0])/3,20, 'obs_id={}'.format(obs_id), fontsize=30)
    
    return ax

def transfer_learning_bootstraps(data, n_trs, n_trs_mist, modelname, stage, n_it=100, start_from_baseline=True, starting_point=None, verbose=True):
    """
    Function to perform transfer learning for any data
    
    Arguments:
        data: list - list of arrays that look like ((logTeff, logLum, logAge, logMass), ...) -  
        n_trs: list - list of integers that are ~about how many stars from each dataset in datas you want in the training sample
        n_trs_mist: int - rough number of stars you want in MIST training set (should generally match n_trs numbers)
        modelname: str - path from this directory to where you want to store the trained model; models will save as 'modelname{}'.format(bootstrap number)
        stage: str - "pre" or "post" (ZAMS); need to add error handeling
        n_it: int - number of iterations to train for (defaults to 100)
        start_from_baseline: bool - whether to start from a baseline model (defaults to True)
        starting_point: str - if start_from_baseline==False, starting_point is the path to the model that will be re-trained instead

    Returns:
        no returns; saves 20 numbered models to the path specified with modelname
    """

    # normalize data
    norm_x_data = []
    norm_y_data = []

    norm_min_preMS = np.load('Aux/norm_min_preMS.npy')
    norm_max_preMS = np.load('Aux/norm_max_preMS.npy')
    
    norm_min_postMS = np.load('Aux/norm_min_postMS.npy')
    norm_max_postMS = np.load('Aux/norm_max_postMS.npy')
    
    for i in range(len(data)):
        dataset=data[i]
        x_data_preMS, x_data_postMS = normalize(dataset)

        y_data_preMS = torch.from_numpy((dataset[:,2:4]-norm_min_preMS[2:])/(norm_max_preMS[2:]-norm_min_preMS[2:])).float()
        y_data_postMS = torch.from_numpy((dataset[:,2:4]-norm_min_postMS[2:])/(norm_max_postMS[2:]-norm_min_postMS[2:])).float()
        
        if stage=="pre":
            norm_x_data.append(x_data_preMS)
            norm_y_data.append(y_data_preMS)
        if stage=="post":
            norm_x_data.append(x_data_postMS)
            norm_y_data.append(y_data_postMS)

            
    # split into train/test
    tr_list = []
    ts_list = []
    for i in range(len(data)):
        x_data = norm_x_data[i]
        y_data = norm_y_data[i]
        n_tr = n_trs[i]

        
        tr_sz = n_tr/len(x_data)
        idx_tr = np.random.rand(x_data.shape[0])<tr_sz

        x_dset_tr = x_data[idx_tr,:]
        y_dset_tr = y_data[idx_tr,:]

        x_dset_ts = x_data[~idx_tr,:]
        y_dset_ts = y_data[~idx_tr,:]

        data_tr = torch.concat((x_dset_tr, y_dset_tr), 1)
        tr_list.append(data_tr)

        data_ts = torch.concat((x_dset_ts, y_dset_ts),1)
        ts_list.append(data_ts)
    
    # add MIST data to training set
    if stage=="pre":
        mist_tl = torch.from_numpy(np.load('Aux/mist_tl_preMS.npy', allow_pickle=True)).float()
    if stage=="post":
        mist_tl = torch.from_numpy(np.load('Aux/mist_tl_postMS.npy',allow_pickle=True)).float()

    tr_sz_mist = n_trs_mist/len(mist_tl)
    idx_tr_mist = np.random.rand(mist_tl.shape[0])<tr_sz_mist
    mist_tl_tr = mist_tl[idx_tr_mist,:]
    mist_tl_ts = mist_tl[~idx_tr_mist,:]

    tr_list.append(mist_tl_tr[:,:4])
    ts_list.append(mist_tl_ts[:,:4])                                 
    
    
    # concatenate training data
    all_tr = torch.cat(tr_list, 0)   
    

    # specify params
    D_in = 2
    D_out = 2
    num_layers = 10
    num_nodes =50
    activation = nn.ReLU() 
    
    # define separate networks for pre/postMS, pre/postMS transfer learning
    net = NN(D_in, D_out, num_layers, num_nodes, activation)
    net_preMS = NN(D_in, D_out, num_layers, num_nodes, activation)
    net_postMS = NN(D_in, D_out, num_layers, num_nodes, activation)
    net_preMS_tl = NN(D_in, D_out, num_layers, num_nodes, activation)
    net_postMS_tl = NN(D_in, D_out, num_layers, num_nodes, activation)

    # define full model:
    num_layers = 10
    num_nodes =50
    activation = nn.ReLU()    
    net = NN(D_in, D_out, num_layers, num_nodes, activation)
    
    # Perform optimization
    loss_fn = nn.MSELoss(reduction='sum')  
    optim = torch.optim.Adam(net.parameters(),lr=1e-3) 
    
    num_iterations = n_it
    
    sz=all_tr.shape[0]

    n_samples=sz
    
    ### THE TRANSFER LEARNING HAPPENS HERE! ###
    for i in range(20):                
        #net.apply(weight_reset)  # don't need since we're starting with a loaded model every time ?
        if start_from_baseline==True:
            if stage=="pre":
                net.load_state_dict(torch.load('Models/Baseline/mist_baseline_preMS10'), strict=False) 
            
            if stage=="post":
                net.load_state_dict(torch.load('Models/Baseline/mist_baseline_postMS10'), strict=False) 

        if start_from_baseline==False:
            net.load_state_dict(torch.load(starting_point), strict=False)
        
        x_data = all_tr[:,:2] 
        y_data = all_tr[:,2:] 

        # the sampling with replacement for this bootstrap
        data_idx = np.random.choice(sz, n_samples)
        x_data = x_data[data_idx]
        y_data = y_data[data_idx]

        
        Loss = []
        if verbose==True:
            print("training model # "+str(i-1))
        for j in range(num_iterations):
            y_data = y_data.reshape(-1,2)   
    
            # run the model forward on the data
            y_pred = net(x_data).squeeze(-1)   # ?
            
            # calculate the mse loss
            loss = loss_fn(y_pred, y_data)
            
            # initialize gradients to zero
            optim.zero_grad()
            
            # backpropagate
            loss.backward() 
            
            # take a gradient step
            optim.step()            
            
            Loss.append(loss)
            if (j + 1) % 10 == 0:
                if verbose==True:
                    print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))
    
        torch.save(net.state_dict(), modelname+'{}'.format(i))   
