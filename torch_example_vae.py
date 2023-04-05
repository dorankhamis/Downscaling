import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import plotnine as p

from torch.distributions.normal import Normal
from torch.distributions import kl
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = 'tight'
plt.rcParams['figure.figsize'] = [8, 4]

run_name = "basic_VAE"  
writer = SummaryWriter("runs_logs/" + run_name)

device = torch.device("cuda") #cuda

BATCH       = 1000 #64 #128 #248
EPOCHS      = 800
LR          = 5e-4
WORKERS     = 2
# rnet_dims   = 128
lat_dim     = 32 #64 128


class BetaBinomial:
    def __init__(self, N, k, m, phi):
        super().__init__()
        self.N   = N
        self.k   = k
        # self.m   = m
        self.phi = phi
        # https://discuss.pytorch.org/t/n-choose-k-function/121974/2
        self.lcoef = torch.lgamma(N + 1) - torch.lgamma(k + 1) - torch.lgamma((N - k) + 1)
        self.m = m.clip(1e-12, 1-1e-12)

    def lbeta_func(self, x, y):
        return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x+y)

    def lprob(self):
        alpha = self.m * self.phi
        beta  = (1-self.m) * self.phi
        return self.lcoef + self.lbeta_func(self.k + alpha, self.N - self.k + beta) - self.lbeta_func(alpha, beta)


BetaBinomial(torch.Tensor([5000]), torch.Tensor([2]), torch.Tensor([1e-4]), torch.Tensor([100])).lprob()


class cycle_VAE(nn.Module):
    def __init__(self, num_genes = 207, dim = 24):
        super().__init__()

        self.encoder = nn.Sequential(
                nn.Linear(num_genes, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.ReLU(),    
                nn.Linear(dim, dim),
                nn.ReLU(), 
                nn.Linear(dim, dim),
                nn.ReLU(),                                
                nn.Linear(dim, 6)
            )

        # self.W    = nn.Parameter(torch.randn(num_genes, 3))
        # self.W    = nn.Parameter(torch.zeros(num_genes, 3))
        self.W    = nn.Parameter(0*torch.ones(num_genes, 3))
        self.lphi = nn.Parameter(1*torch.ones(num_genes))

    def predict(self, x):
        vae_params = self.encoder(x.to(torch.float32)) # 6 dims are: radius, theta, z3 mean and var
        r   = vae_params[:,0].exp()
        thta = vae_params[:,2]
        z3    = vae_params[:,4]
        z_mat = torch.stack([r * thta.cos(), r * thta.sin(), z3], dim = 1)
        return thta, z_mat


    def forward(self, x, read_dpth):
        vae_params = self.encoder(x.to(torch.float32)) # 6 dims are: radius, theta, z3 mean and var

        # priors on variational params
        ones_vec = torch.ones_like(read_dpth.squeeze())
        prior_rad   = Normal(ones_vec, 0.1*ones_vec) 
        prior_theta = Normal(torch.pi*ones_vec, 4*torch.pi*ones_vec) 
        prior_z3    = Normal(-5*ones_vec, 0.5*ones_vec) 
        
        # variational normal dists 
        norm_rad   = Normal(vae_params[:,0].exp(), vae_params[:,1].exp()) 
        norm_theta = Normal(vae_params[:,2], vae_params[:,3].exp()) 
        norm_z3    = Normal(vae_params[:,4], vae_params[:,5].exp()) 

        # make z mat == 
        r     = norm_rad.rsample()
        thta  = norm_theta.rsample()
        z3    = norm_z3.rsample()
        z_mat = torch.stack([r * thta.cos(), r * thta.sin(), z3], dim = 1)

        # Gene proportions
        props = torch.matmul(z_mat, self.W.t()).sigmoid()
        # print(props)
        exp_log_lik = BetaBinomial(read_dpth.unsqueeze(1), x, props, self.lphi.exp()).lprob().sum()
        # print(exp_log_lik)

        KL = kl.kl_divergence(norm_rad, prior_rad) + kl.kl_divergence(norm_theta, prior_theta) + kl.kl_divergence(norm_z3, prior_z3)
        
        ELBO_i = exp_log_lik - KL.sum() 
        # return z, prop, ELBO_i, KL
        return ELBO_i, KL


net     = cycle_VAE().to(device)
tot_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("{:5.2f} million params".format(tot_params*1e-6))
ELBO_i, KL = net(torch.randint(2, 100, [100, 207], device = device), 2000*torch.ones([100,1], device = device))
print((ELBO_i.shape, KL.shape))


def make_predictions(net, dat_loader):
    # we need to make sure the network is in the evaluation mode not training
    net.eval()
    z_mat_test = []
    thta_test =  []
    # we have this command because each time we run the computational graph, it updates the gradiants.
    # we want to make sure not to update the gradiants here.
    with torch.no_grad():
        for i, (inputs, read_dpth) in enumerate(dat_loader, 0):
            inputs, read_dpth = inputs.to(device), read_dpth.to(device)
            thta, z_mat = net.predict(inputs)
            thta = thta.cpu().numpy()
            thta = np.arctan2(np.sin(thta), np.cos(thta))
            thta_test.append(thta)
            z_mat_test.append(z_mat.cpu().numpy())
    return np.concatenate(thta_test), np.concatenate(z_mat_test)
    
    
training_data   = data_utils.TensorDataset(counts_x, read_dpth)

train_loader    = DataLoader(training_data, batch_size=BATCH, shuffle=True, num_workers=WORKERS, pin_memory=True)
val_loader      = DataLoader(training_data, batch_size=BATCH, shuffle=False, num_workers=WORKERS, pin_memory=True)
len(train_loader)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)


iter_i = 0 
net = net.to(device)
for epoch in range(EPOCHS):  # loop over the dataset multiple times

    running_elbo = []
    # indx = 0
    for i, (inputs, read_dpth) in enumerate(train_loader, 0):
        inputs, read_dpth = inputs.to(device), read_dpth.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # Warm up weight of Kl part
        ELBO_i, KL = net(inputs, read_dpth)
        elb_mean_neg = -ELBO_i/BATCH
        # print(elb_mean_neg.shape)
        elb_mean_neg.backward()
        optimizer.step()

        # print statistics
        running_elbo.append(elb_mean_neg.item())
        
    print(f'[Epoch {epoch + 1}] loss: {np.mean(running_elbo):.6f}')
    writer.add_scalar('Epoch_Loss/neg_ELBO', np.mean(running_elbo), epoch)

print('Finished Training')

    
