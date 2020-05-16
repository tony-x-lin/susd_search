import torch
import torch.nn as nn
from torch.optim import Adam

import numpy as np
import os

criterion = nn.MSELoss()

class Dynamics(nn.Module):
    def __init__(self, nb_states, nb_actions):
        super(Dynamics, self).__init__()
        self.f1 = nn.Linear(nb_states+nb_actions, 100)
        self.f2 = nn.Linear(100, 80)
        self.f3 = nn.Linear(80, 60)
        self.f4 = nn.Linear(60, 40)
        self.f5 = nn.Linear(40, nb_states)

        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.sig(self.f1(x))
        y = self.sig(self.f2(y))
        y = self.sig(self.f3(y))
        y = self.sig(self.f4(y))

        return self.f5(y)

class BB_Model:
    def __init__(self, nb_states, nb_actions, train=True, use_cuda=False, lr=0.001):
        self.model = Dynamics(nb_states, nb_actions).double()
        self.model_optim = Adam(self.model.parameters(), lr=lr)

        if train: self.model.train()
        else: self.model.eval()

        if use_cuda: self.model.cuda(); self.cuda = True
        else: self.model.cpu(); self.cuda = False

    def predict(self, x, u):
        """ x is a vertical stack of [x; u]"""
        return self.model(torch.from_numpy(np.vstack((x, u)).T)).detach().numpy().T

    def train(self, x0, u, y):
        # get forward pass
        yhat = self.model(torch.from_numpy(np.vstack((x0, u)).T))
        
        # perform backward pass
        self.model_optim.zero_grad()
        loss = criterion(torch.from_numpy(y.T), yhat)
        loss.backward()
        self.model_optim.step()
        
    def save_model(self, path=''):
        """
        Save the current model at the given path in the folder 'weights' with pickle.
        Example: path='test', saved file will be at weights/testmodel.pkl
        """
        if not os.path.exists('weights'):
            os.makedirs('weights')
            
        torch.save(self.model.state_dict(), 'weights/'+path+'model.pkl')
        
    def load_model(self, path=''):
        """
        Load the model from the given path in the folder 'weights' with pickle.
        Example: path='test', loaded file will be at weights/testmodel.pkl
        """
        if os.path.exists('weights/'+path+'model.pkl'):
            if self.cuda:
                self.model.load_state_dict(torch.load('weights/'+path+'model.pkl'))
            else:            
                self.model.load_state_dict(torch.load('weights/'+path+'model.pkl', map_location='cpu'))
            return True

        return False
