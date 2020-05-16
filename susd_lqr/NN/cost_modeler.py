import torch
import torch.nn as nn
from torch.optim import Adam

import numpy as np
import os

criterion = nn.MSELoss()

class Cost(nn.Module):
    def __init__(self, nb_states, nb_gains):
        super(Cost, self).__init__()
        self.f1 = nn.Linear(nb_states+nb_gains, 250)
        self.f2 = nn.Linear(250, 200)
        self.f3 = nn.Linear(200, 150)
        self.f4 = nn.Linear(150, 100)
        self.f5 = nn.Linear(100, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.f1(x))
        y = self.relu(self.f2(y))
        y = self.relu(self.f3(y))
        y = self.relu(self.f4(y))

        return self.f5(y)

class BB_Cost_Model:
    def __init__(self, nb_states, nb_gains, train=True, use_cuda=False, lr=0.001):
        self.model = Cost(nb_states, nb_gains).double()
        self.model_optim = Adam(self.model.parameters(), lr=lr)

        if train: self.model.train()
        else: self.model.eval()

        if use_cuda: self.model.cuda()
        else: self.model.cpu()

    def predict(self, x0, k):
        """ x is a vertical stack of [x; u]"""
        return self.model(torch.from_numpy(np.vstack((x0, k)).T)).detach().numpy().T

    def train(self, x0, k, z):
        # get forward pass
        zhat = self.model(torch.from_numpy(np.vstack((x0, k)).T))
        
        # perform backward pass
        self.model_optim.zero_grad()
        loss = criterion(torch.from_numpy(z.T), zhat)
        loss.backward()
        self.model_optim.step()
        
    def save_model(self, path=''):
        """
        Save the current model at the given path in the folder 'weights' with pickle.
        Example: path='test', saved file will be at weights/testmodel.pkl
        """
        if not os.path.exists('weights'):
            os.makedirs('weights')
            
        torch.save(self.model.state_dict(), 'weights/'+path+'cost.pkl')
        
    def load_model(self, path=''):
        """
        Load the model from the given path in the folder 'weights' with pickle.
        Example: path='test', loaded file will be at weights/testmodel.pkl
        """
        if os.path.exists('weights/'+path+'cost.pkl'):
            self.model.load_state_dict(torch.load('weights/'+path+'cost.pkl'))
            return True

        return False
