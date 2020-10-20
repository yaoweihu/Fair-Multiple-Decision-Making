import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.unconstrained_model import UnconstrainedModel

torch.manual_seed(1)


class JointModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.tau = kwargs['tau']
        self.choose_phi(kwargs['phi_name'])
        self.prob_dict = self.to_tensor(kwargs['prob_dict'])

        self.w1 = nn.Parameter(torch.randn(0))
        self.w2 = nn.Parameter(torch.randn(0))

        self.lmd1 = torch.tensor(kwargs['lmd1'])
        self.lmd2 = torch.tensor(kwargs['lmd2'])

    def to_tensor(self, prob_dict):
        res = {}
        for key, val in prob_dict.items():
            res[key] = torch.tensor(val.values, dtype=torch.float32)
        return res

    def choose_phi(self, phi_name):
        if phi_name == 'logistic':
            self.cvx_phi = lambda z: torch.log(1 + torch.exp(-z))
        elif phi_name == 'hinge':
            self.cvx_phi = lambda z: torch.relu(1.0 - z)
        elif phi_name == 'exponential':
            self.cvx_phi = lambda z: torch.exp(-z)
        else:
            print("Your surrogate function doesn't exist.")

    def init_weights(self, x1, x2):
        self.w1 = nn.Parameter(torch.randn(x1.shape[1]))
        self.w2 = nn.Parameter(torch.randn(x2.shape[1]))
        
    def forward(self, x1, x2, iter):
        x1 = torch.tensor(x1.values, dtype=torch.float32)
        h1 = x1 @ self.w1
        R1 = torch.mean(self.cvx_phi(h1) * self.prob_dict['R1_1'] + self.cvx_phi(-h1) * self.prob_dict['R1_0'])
        T1 = torch.mean(self.cvx_phi(-h1) * self.prob_dict['T1_1'] + self.cvx_phi(h1) * self.prob_dict['T1_0'])

        x2 = torch.tensor(x2.values, dtype=torch.float32)
        x20, x21 = x2.clone(), x2.clone()
        x20[:, 0] = 0
        x21[:, 0] = 1
        h20 = x20 @ self.w2
        h21 = x21 @ self.w2        
       
        R2 = torch.mean(self.cvx_phi(h21) * self.cvx_phi(-h1) * self.prob_dict['R2_11'] + 
                        self.cvx_phi(h20) * self.cvx_phi(h1) * self.prob_dict['R2_01'] + 
                        self.cvx_phi(-h21) * self.cvx_phi(-h1) * self.prob_dict['R2_10'] + 
                        self.cvx_phi(-h20) * self.cvx_phi(h1) * self.prob_dict['R2_00'])

        T2 = torch.mean(self.cvx_phi(-h21) * self.cvx_phi(-h1) * self.prob_dict['T2_11'] + 
                        self.cvx_phi(-h20) * self.cvx_phi(h1) * self.prob_dict['T2_10'] + 
                        self.cvx_phi(h21) * self.cvx_phi(-h1) * self.prob_dict['T2_01'] + 
                        self.cvx_phi(h20) * self.cvx_phi(h1) * self.prob_dict['T2_00'])
        
        loss = R1 + R2 + self.lmd1 * torch.relu((T1 - 1) - self.tau[0]) + self.lmd2 * torch.relu((T2 - 1) - self.tau[1])# + \
                         #self.lmd1 * torch.relu(-self.tau[0] - (1 - T1)) + self.lmd2 * torch.relu(-self.tau[1] - (1 - T2))
    
        # if iter % 1000 == 0:
        #     print(f"iter: {iter} loss: {loss.item():0.3f}, R1: {R1.item():0.3f}, R2: {R2.item():0.3f}, "
        #           f"T1: {T1.item()-1:0.3f}, LT1: {(self.lmd1 * torch.relu((T1 - 1) - self.tau[0])).item():0.3f}, "
        #           f"T2: {T2.item()-1:0.3f}, LT2: {(self.lmd2 * torch.relu((T2 - 1) - self.tau[1])).item():0.3f}")
        return loss

    def acc_fair(self, x1, x2):
        x1 = torch.tensor(x1.values, dtype=torch.float32)
        h1 = x1 @ self.w1
        Acc1 = torch.mean((h1 >= 0).type(torch.FloatTensor) * self.prob_dict['R1_1'] + (h1 < 0).type(torch.FloatTensor) * self.prob_dict['R1_0'])
        Fair1 = torch.mean((h1 >= 0).type(torch.FloatTensor) * self.prob_dict['T1_1'] + (h1 < 0).type(torch.FloatTensor) * self.prob_dict['T1_0']) - 1

        x2 = torch.tensor(x2.values, dtype=torch.float32)
        x20, x21 = x2.clone(), x2.clone()
        x20[:, 0] = 0
        x21[:, 0] = 1

        h20 = x20 @ self.w2
        h21 = x21 @ self.w2

        Acc2 = torch.mean((h21 >= 0).type(torch.FloatTensor) * (h1 < 0).type(torch.FloatTensor) * self.prob_dict['R2_11'] + \
                          (h20 >= 0).type(torch.FloatTensor) * (h1 >= 0).type(torch.FloatTensor) * self.prob_dict['R2_01'] + \
                          (h21 < 0).type(torch.FloatTensor) * (h1 < 0).type(torch.FloatTensor) * self.prob_dict['R2_10'] + \
                          (h20 < 0).type(torch.FloatTensor) * (h1 >= 0).type(torch.FloatTensor) * self.prob_dict['R2_00'])
        Fair2 = torch.mean((h21 >= 0).type(torch.FloatTensor) * (h1 >= 0).type(torch.FloatTensor) * self.prob_dict['T2_11'] + \
                           (h20 >= 0).type(torch.FloatTensor) * (h1 < 0).type(torch.FloatTensor) * self.prob_dict['T2_10'] + \
                           (h21 < 0).type(torch.FloatTensor) * (h1 >= 0).type(torch.FloatTensor) * self.prob_dict['T2_01'] + \
                           (h20 < 0).type(torch.FloatTensor) * (h1 < 0).type(torch.FloatTensor) * self.prob_dict['T2_00']) - 1

        return [Acc1.item(), Fair1.item()], [Acc2.item(), Fair2.item()]


class JointConstrainedModel(UnconstrainedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_name = kwargs['name']
        self.generate_eval_data = kwargs['generate_eval_data']
        self.nnmodel = JointModel(**kwargs)
        self.iters = kwargs['iters']  
        self.lr = kwargs['lr']
        self.w1 = 0
        self.w2 = 0

    def fit(self, s, x1, x2, **kwargs):
        kwargs['dummy_cols'] = ['x1']
        x1 = self.pre_processing(s, x1, **kwargs)
    
        if self.data_name == 'synthetic':
            kwargs['dummy_cols'] = ['x2']
        x2 = self.pre_processing(s, x2, **kwargs)
        self.nnmodel.init_weights(x1, x2)

        optimizer = torch.optim.Adam(self.nnmodel.parameters(), lr=self.lr)
        for i in range(self.iters + 1):
            loss = self.nnmodel(x1, x2, i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        res = self.nnmodel.acc_fair(x1, x2)
        
        self.w1 = self.nnmodel.w1.detach().numpy()
        self.w2 = self.nnmodel.w2.detach().numpy()

        return res