import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MoCo(torch.nn.modules.loss._Loss):
    def __init__(self, device, T=0.5):

        super(MoCo, self).__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive, queue):
        
        emb_anchor = torch.mm(torch.diag(torch.sum(torch.pow(emb_anchor, 2), axis=1) ** (-0.5)), emb_anchor)
        emb_positive = torch.mm(torch.diag(torch.sum(torch.pow(emb_positive, 2), axis=1) ** (-0.5)), emb_positive)
        queue = torch.mm(torch.diag(torch.sum(torch.pow(queue, 2), axis=1) ** (-0.5)), queue)

        l_pos = torch.einsum('nc,nc->n', [emb_anchor, emb_positive]).unsqueeze(-1)
        l_neg = torch.einsum('nc,kc->nk', [emb_anchor, queue])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        loss = F.cross_entropy(logits, labels)
        
        return loss


class BYOL(torch.nn.modules.loss._Loss):

    def __init__(self, device, T=0.5):
   
        super(BYOL, self).__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive):

        # L2 normalize
        emb_anchor = torch.mm(torch.diag(torch.sum(torch.pow(emb_anchor, 2), axis=1) ** (-0.5)), emb_anchor)
        emb_positive = torch.mm(torch.diag(torch.sum(torch.pow(emb_positive, 2), axis=1) ** (-0.5)), emb_positive)

        l_pos = torch.einsum('nc,nc->n', [emb_anchor, emb_positive]).unsqueeze(-1)
        l_neg = torch.mm(emb_anchor, emb_positive.t())

        loss = - l_pos.sum()
                
        return loss


class SimSiam(torch.nn.modules.loss._Loss):

    def __init__(self, device, T=0.5):
    
        super(SimSiam, self).__init__()
        self.T = T
        self.device = device

    def forward(self, p1, p2, z1, z2):

        # L2 normalize
        p1 = F.normalize(p1, p=2, dim=1)
        p2 = F.normalize(p2, p=2, dim=1)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # mutual prediction
        l_pos1 = torch.einsum('nc,nc->n', [p1, z2.detach()]).unsqueeze(-1)
        l_pos2 = torch.einsum('nc,nc->n', [p2, z1.detach()]).unsqueeze(-1)

        loss = - (l_pos1.sum() + l_pos2.sum())
                
        return loss


import torch
import torch.nn.functional as F

class OurLoss(torch.nn.modules.loss._Loss):

    def __init__(self, device, margin=0.5, sigma=2.0, T=2.0, kernel_bandwidth=0.5, num_kernels=3):
        super(OurLoss, self).__init__()
        self.device = device
        self.margin = margin
        self.sigma = sigma
        self.T = T
        self.kernel_bw = kernel_bandwidth
        
        self.num_kernels = num_kernels  

    def forward(self, emb_anchor, emb_positive):
        
        emb_anchor = F.normalize(emb_anchor, p=2, dim=1)
        emb_positive = F.normalize(emb_positive, p=2, dim=1)

        sim = torch.mm(emb_anchor, emb_positive.t()) / self.T

        
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim_mean = torch.mean(sim, dim=1, keepdim=True)

        total_weights = torch.zeros_like(sim)

       
        
        if self.num_kernels > 1:
  
            step = (sim_max - sim_mean) / (self.num_kernels - 1 + 1e-6)
            
            for i in range(self.num_kernels):
              
                current_mu = sim_mean + i * step
                
               
                w_i = torch.exp(-torch.pow(sim - current_mu, 2) / (2 * self.kernel_bw ** 2))
                
                total_weights += w_i
        else:
            total_weights = torch.exp(-torch.pow(sim - sim_max, 2) / (2 * self.kernel_bw ** 2))

       
        total_weights = total_weights / (torch.sum(total_weights, dim=1, keepdim=True) + 1e-8)

        neg = torch.mm(total_weights, emb_positive)

       
        l_pos = torch.exp(-torch.sum(torch.pow(emb_anchor - emb_positive, 2), dim=1) / (2 * self.sigma ** 2))
        l_neg = torch.exp(-torch.sum(torch.pow(emb_anchor - neg, 2), dim=1) / (2 * self.sigma ** 2))

        zero_matrix = torch.zeros(l_pos.shape).to(self.device)
        loss = torch.max(zero_matrix, l_neg - l_pos + self.margin).mean()
        
        return loss
