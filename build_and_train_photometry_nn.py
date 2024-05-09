# VERSION OF TRAINING SCRIPT WITH NO SCHEDULER USED !!!

model_name = "payne_10layer_10node_1000it_1em3lr_example_for_github"
print("running script to train:", model_name)
# imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# seed = 42
# torch.manual_seed(seed) 
import sys
from astropy.table import Table
import matplotlib.pyplot as plt

print("done importing packages")

######################################
######## UPDATE THESE BEFORE RUNNING #######
# sys.path.append("update the path to access payne dataset in another directory")
full_dataset=False # if true, uses full Payne dataset (too large to upload to GitHub, available upon request)
num_layers = 10
num_nodes = 10
learning_rate = 1e-3
num_iterations = 1000 
activation = nn.ReLU()

# read in training data
print("reading in data")

if full_dataset==True:
    print("full payne dataset too large for github repository but available upon request!")
    # datapath = # ADD PATH TO FULL DATASET gaiaPredPhot.fits HERE 
    # #payne = Table.read(datapath+"GaiaPredPhot.fits", format="fits")
    # #payne_df = payne.to_pandas()
    # print(" done reading data")

if full_dataset==False:
    payne_df = pd.read_csv("payne_example_dataset.csv")
    payne_df = payne_df[["GaiaDR3_G","GaiaDR3_BP","GaiaDR3_RP","logTeff","logLum"]].copy()
    print(" done reading data")



print("normalizing data, converting to torch tensor")

params = ["GaiaDR3_G","GaiaDR3_BP", "GaiaDR3_RP", "logTeff", "logLum"]

# note--realizing it might not be correct practice to take the min and max of the full dataset
###--possibly should be using the min/max of the training data only?
norm_min = np.load("Aux/norm_min_payne.npy", allow_pickle=True)
norm_max = np.load("Aux/norm_max_payne.npy", allow_pickle=True)


payne_data = payne_df[["GaiaDR3_G", "GaiaDR3_BP", "GaiaDR3_RP", "logTeff","logLum"]].values.astype(float)

def normalize(data):
    """
    data must be a 2d array that looks like
    [[M_G, M_BP, M_RP, logTeff, logLum],...]
    """
    data_norm = (data - norm_min) / (norm_max - norm_min)

    return data_norm

def unnormalize(data_norm):
    data_un = data_norm*(norm_max - norm_min) + norm_min
    return data_un

# normalize all payne data
payne_norm = normalize(payne_data)

# convert to torch tensor
payne_norm = torch.from_numpy(payne_norm)
print(" done")

print("train/test split")
tr_sz = 0.8
idx_tr = np.random.rand(payne_norm.shape[0])<tr_sz

payne_tr = payne_norm[idx_tr,:]
payne_ts = payne_norm[~idx_tr,:]
print("training shape:", payne_tr.shape)
print("test shape:", payne_ts.shape)
print(" done")



# save data/info for testing in a jupyter notebook
print("saving train/test split for reference")
np.save("Models/Photometry/loss_hyperparams_trts_info/"+model_name+"_payne_tr.npy", payne_tr.numpy(), allow_pickle=True)
np.save("Models/Photometry/loss_hyperparams_trts_info/"+model_name+"_payne_ts.npy", payne_ts.numpy(), allow_pickle=True)
print(" done saving train/test split")



######################### MODEL BUILDING TIME #####################################
print("writing NN class")
class NN(nn.Module):
    def __init__(self, D_in, D_out, num_layers, num_nodes, activation):
        super(NN, self).__init__()  # idk what this does

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

print("specifying hyperparams:")
D_in = 3   # M_G, M_BP, M_RP
D_out = 2  # logTeff, logLum

net = NN(D_in, D_out, num_layers, num_nodes, activation)

print("D_in = ", D_in)
print("D_out = ", D_out)
print("num_layers = ", num_layers)
print("num_nodes = ", num_nodes)
print("learning rate = ", learning_rate)
print("n iterations = ", num_iterations)
print("activation = relu")

# split x and y training data, make them the right kind of float
x_data = payne_tr[:,:3].to(torch.float32)
y_data = payne_tr[:,3:].to(torch.float32)

x_data_test = payne_ts[:,:3].to(torch.float32) 
y_data_test = payne_ts[:,3:].to(torch.float32)

print("training:")
# define loss function and optimizer
loss_fn = nn.MSELoss(reduction='sum')
optim = torch.optim.Adam(net.parameters(), lr=learning_rate) 

Loss = []
Loss_test = []

for j in range(num_iterations):

    y_data.reshape(-1,2)
    y_data_test.reshape(-1,2)
    # run model forward on the data:
    y_pred = net(x_data).squeeze(-1) # idk what the squeeze is
    y_pred_test = net(x_data_test).squeeze(-1)
    # calculate mse loss
    loss = loss_fn(y_pred, y_data)
    loss_test = loss_fn(y_pred_test, y_data_test)
    # initialize gradients to 0
    optim.zero_grad()

    # backpropogate
    loss.backward()

    # take gradient step
    optim.step()

    Loss.append(loss)
    Loss_test.append(loss_test)
    
    if (j+1)%5 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))

print("saving model")    
torch.save(net.state_dict(), "Models/Photometry/"+model_name)

print("done training")


###############################################################################
# save a file with the loss info & hyperparams
print("saving loss and hyperparam info")
# loss dataframe
loss_df = pd.DataFrame()
Loss_train_list = [float(loss.detach().numpy()) for loss in Loss] # make tensors regular arrays >:(
Loss_test_list = [float(loss.detach().numpy()) for loss in Loss_test]
loss_df["loss_train"] = Loss_train_list
loss_df["loss_test"] = Loss_test_list
loss_df["iteration"] = loss_df.index + 1

# throw in the hyperparams what the heck
loss_df["n_layers"] = [num_layers]*len(loss_df)
loss_df["n_nodes"] = [num_nodes]*len(loss_df)
loss_df["learning_rate"] = [learning_rate]*len(loss_df)

# model_name
loss_df.to_csv("Models/Photometry/loss_hyperparams_trts_info/"+model_name+"_hyperparams_loss.csv")
print("saved")

print("done running yay!")