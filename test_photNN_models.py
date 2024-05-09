import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
plt.rcParams.update({
    'figure.figsize':(7,7),
    'xtick.major.width':1,
    'ytick.major.width':1,
    'xtick.minor.width':1,
    'ytick.minor.width':1,
    'xtick.major.size':8,
    'ytick.major.size': 6,
    'xtick.minor.size':3,
    'ytick.minor.size':3,
    'font.family':'STIXGeneral',
    'font.size':16,
#    'xtick.top':True,
#    'ytick.right':True,
    'xtick.direction':'in',
    'ytick.direction':'in',
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif'}) 


class test_photnn():
    """
    automated testing for new StelNet models

    subroutines:
    __init__() - load data
    NN() - torch class for building models
    model_load_predict() - 
    """
    def __init__(self, modelname, n_layers, n_nodes):
        """
        initialize class--initializes by loading normalized data

        load normalized training and testing data/make an attribute for it

        modelname: str--name assigned during training with OSC
        n_layers: int--number of layers for this model
        n_nodes: int--number of nodes for this model
        test_data: if none, uses the stuff that came from the script running - should be a pkl file that goes [Mg, Mbp, Mrp, log(T/K), log(L/Lsun)]

        attributes: 
        modelname - string the modelnme
        n_layers/n_nodes - number of layers/nodes input to be used for loading models later
        train_data - [Mg, Mbp, Mrp, teff, lum] torch tensor
        x_train/y_train - photometry torch tensor/teff lum torch tensor
        ^^ same for test
        """

        # load training and testing data and save as class attributes        
        self.modelname=modelname
        train_data = np.load("Models/Photometry/loss_hyperparams_trts_info/"+modelname+"_payne_tr.npy",
                             allow_pickle=True)
        self.train_data = torch.from_numpy(train_data)
        self.x_train = self.train_data[:,:3].to(torch.float32)
        self.y_train = self.train_data[:,3:].to(torch.float32)
        
        test_data = np.load("Models/Photometry/loss_hyperparams_trts_info/"+modelname+"_payne_ts.npy",
                             allow_pickle=True) 
        self.test_data = torch.from_numpy(test_data)
        self.x_test = self.test_data[:,:3].to(torch.float32)
        self.y_test = self.test_data[:,3:].to(torch.float32) 


        self.norm_min = np.load("Aux/norm_min_payne.npy", allow_pickle=True)
        self.norm_max = np.load("Aux/norm_max_payne.npy", allow_pickle=True)

        self.n_layers = n_layers
        self.n_nodes = n_nodes
   
    # inner class so that we can make the models
    class NN(nn.Module):
        def __init__(self, D_in, D_out, num_layers, num_nodes, activation):
            super().__init__()#NN, self).__init__()  # idk what this does
    
            # specify list of layer sizes
            sizes = [D_in] + [num_nodes] * num_layers + [D_out]
            in_sizes, out_sizes = sizes[:-1], sizes[1:]  # all but output, all but input
    
            # construct linear layers
            self.linears = nn.ModuleList() # list of all xformations between layers (rn is empty)
            for n_in, n_out in zip(in_sizes, out_sizes):
                self.linears.append(nn.Linear(n_in, n_out)) # append a linear xformation to each..transition between layers
    
            # specify activation function
            self.activation = activation
    
        # forward pass of the network
        def forward(self, x):
            
            for l in self.linears[:-1]:     # ie for everything but the last hidden layer -> output transformation
                x = self.activation(l(x))   # take the activation function of the corresponding linear xformation
            x = self.linears[-1](x)         # for the last jump do not use the activation function for some reason idk
    
            return x

    def model_load_predict(self):
        """
        loads a network (model name) with num_layers layers and num_nodes nodes/layer (see model naming convention)
        and predicts for x_train and x_test to return y_pred_train, y_pred_test

        saves train/test predictions and residuals as class attributes
        """
        D_in = 3   # M_G, M_BP, M_RP
        D_out = 2  # logTeff, logLum
        activation = nn.ReLU()
        
        net = self.NN(D_in, D_out, self.n_layers, self.n_nodes, activation)
        net.load_state_dict(torch.load("Models/Photometry/"+self.modelname),
                            strict=False)
        y_pred_train = torch.unsqueeze(net(self.x_train),0).detach().numpy()
        y_pred_test = torch.unsqueeze(net(self.x_test),0).detach().numpy()
    
        self.y_pred_test = y_pred_test[0] #reshape
        self.y_pred_train = y_pred_train[0]

        y_train_np = self.y_train.detach().numpy()
        y_test_np = self.y_test.detach().numpy()
        
        self.residuals_train = (10**self.y_pred_train) - (10**y_train_np)   # residuals not in log space!
        self.residuals_test = (10**self.y_pred_test) - (10**y_test_np)     # no sir not in log space >:)
    

    # unnormalize stuff
    def unnormalize_train_test(self):
        """
        add un-normalized outputs as class attributes
        """
        self.y_pred_train_un = self.y_pred_train*(self.norm_max[3:] - self.norm_min[3:]) + self.norm_min[3:]
        self.y_pred_test_un = self.y_pred_test*(self.norm_max[3:] - self.norm_min[3:]) + self.norm_min[3:]

        self.y_train_un = self.y_train*(self.norm_max[3:] - self.norm_min[3:]) + self.norm_min[3:]
        self.y_test_un = self.y_test*(self.norm_max[3:] - self.norm_min[3:]) + self.norm_min[3:]

        y_test_un_np = self.y_test_un.detach().numpy()        
        self.residuals_test_un = (10**self.y_pred_test_un) - (10**y_test_un_np)  # residuals not in log space!
        self.residuals_test_percent_un = np.abs(self.residuals_test_un) / 10**(self.y_test_un)
        

    def plot_performance(self, figname=None):
        """
        plot predicted vs true value for teff and luminosity
        preferably both un-normalized
        """
        fig, [ax1,ax2] = plt.subplots(1,2, figsize=[10,5])
        axs=[ax1,ax2]
        ax1.scatter(self.y_test_un[:,:1], self.y_pred_test_un[:,:1], s=5, rasterized=True)
        ax2.scatter(self.y_test_un[:,1:], self.y_pred_test_un[:,1:], s=5, rasterized=True)
        
        for ax in [ax1,ax2]:
            ax.set_xlabel("true")
            ax.set_ylabel("predicted")
        #    ax.plot([0,1],[0,1], c='r')
        
        ax1.plot([3.4,4],[3.4,4],c='r')
        ax2.plot([-3,5],[-3,5],c='r')
        ax1.set_title("log(teff/k)")
        ax2.set_title("log(L/Lsun)")


        plt.show()

    def plot_loss(self, mean_MSE=True, check_tail=False, figname=None):
        self.loss_df = pd.read_csv(
            "Models/Photometry/loss_hyperparams_trts_info/"+self.modelname+"_hyperparams_loss.csv", comment="#"
        )
        if mean_MSE==True:
            self.loss_df["mean_loss_train"] = self.loss_df["loss_train"]/len(self.train_data)
            self.loss_df["mean_loss_test"] = self.loss_df["loss_test"]/len(self.test_data)
            fig, ax = plt.subplots(figsize=[5,4])
            ax.plot(self.loss_df["iteration"], self.loss_df["mean_loss_train"], label="training data")
            ax.plot(self.loss_df["iteration"], self.loss_df["mean_loss_test"], label="test data")
            ax.legend(loc="upper right")
            ax.set_xlabel("iteration")
            ax.set_ylabel("mean loss")
            ax.set_yscale("log")

        if check_tail==True: # only look at last 200 iterations
            ax.set_xlim(len(self.loss_df)-200, len(self.loss_df))
    
        if mean_MSE==False:
            fig, ax = plt.subplots(figsize=[5,4])
            
            ax.plot(self.loss_df["iteration"], self.loss_df["loss"])
            ax.set_xlabel("iteration")
            ax.set_ylabel("summed loss")
            ax.set_yscale("log")      

        plt.show()



    def plot_outputspace_residuals(self, percent_residual=True, figname=None):
        test_un = self.y_test_un
        pred_test_un = self.y_pred_test_un
        
        if percent_residual==False:
            self.residuals_teff = np.abs(self.residuals_test_un[:,0])
            self.residuals_lum = np.abs(self.residuals_test_un[:,1])

        if percent_residual==True:
            self.residuals_teff = np.abs(self.residuals_test_un[:,0])/(10**self.y_test_un[:,0])
            self.residuals_lum = np.abs(self.residuals_test_un[:,1])/(10**self.y_test_un[:,1])
       
        fig, ax = plt.subplots()
        
        # plot luminosity vs temperature, color by normed residual
        ob=ax.scatter(test_un[:,0], test_un[:,1], c=self.residuals_teff, s=30, cmap="copper", rasterized=True)
        divider=make_axes_locatable(ax)
        cax = divider.append_axes("right",size="5%", pad=0)

        if percent_residual==True:
            colorbar = fig.colorbar(ob, cax=cax, label="|% Teff residual|")
        
        if percent_residual==False:
            colorbar = fig.colorbar(ob, cax=cax, label="|Teff residual|")
        
        ax.invert_xaxis()
        ax.set_xlabel("log(Teff/K)")
        ax.set_ylabel("log(L/Lsun)")

        
        fig, ax = plt.subplots()
        
        # plot luminosity vs temperature, color by normed residual
        ob=ax.scatter(test_un[:,0], test_un[:,1], c=self.residuals_lum, s=30, cmap="copper", rasterized=True)
        divider=make_axes_locatable(ax)
        cax = divider.append_axes("right",size="5%", pad=0)

        if percent_residual==True:
            colorbar = fig.colorbar(ob, cax=cax, label="|% lum residual|")
        if percent_residual==False:
            colorbar = fig.colorbar(ob, cax=cax, label="|lum residual|")
    
        ax.invert_xaxis()
        ax.set_xlabel("log(Teff/K)")
        ax.set_ylabel("log(L/Lsun)")  